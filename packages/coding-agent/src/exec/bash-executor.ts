/**
 * Bash command execution with streaming support and cancellation.
 *
 * Uses brush-core via native bindings for shell execution.
 */
import * as fs from "node:fs/promises";
import { executeShell, type MinimizerOptions, Shell } from "@oh-my-pi/pi-natives";
import { Settings, type ShellMinimizerSettings } from "../config/settings";
import { OutputSink } from "../session/streaming-output";
import { resolveOutputMaxColumns, resolveOutputSinkHeadBytes } from "../tools/output-meta";
import { getOrCreateSnapshot } from "../utils/shell-snapshot";
import { NON_INTERACTIVE_ENV } from "./non-interactive-env";

export interface BashExecutorOptions {
	cwd?: string;
	timeout?: number;
	onChunk?: (chunk: string) => void;
	signal?: AbortSignal;
	/** Session key suffix to isolate shell sessions per agent */
	sessionKey?: string;
	/** Additional environment variables to inject */
	env?: Record<string, string>;
	/** Artifact path/id for full output storage */
	artifactPath?: string;
	artifactId?: string;
	/**
	 * Invoked when the native minimizer rewrote the command's output, giving
	 * the caller a chance to persist the lossless original capture (typically
	 * via the session's `ArtifactManager`). The returned id is spliced into
	 * the sink output as `artifact://<id>` so the agent can retrieve the raw
	 * bytes. Return `undefined` to skip the footer.
	 */
	onMinimizedSave?: (
		originalText: string,
		info: { filter: string; inputBytes: number; outputBytes: number },
	) => Promise<string | undefined>;
}

export interface BashResult {
	output: string;
	exitCode: number | undefined;
	cancelled: boolean;
	truncated: boolean;
	totalLines: number;
	totalBytes: number;
	outputLines: number;
	outputBytes: number;
	artifactId?: string;
}

const HARD_TIMEOUT_GRACE_MS = 5_000;

const shellSessions = new Map<string, Shell>();
const brokenShellSessions = new Set<string>();

async function resolveShellCwd(cwd: string | undefined): Promise<string | undefined> {
	if (!cwd) return undefined;

	try {
		// Brush preserves the working directory string verbatim, so resolve symlinks
		// up front to keep `pwd` aligned with tools like `git worktree list`.
		return await fs.realpath(cwd);
	} catch {
		return cwd;
	}
}

/** Translate `ShellMinimizerSettings` into native `MinimizerOptions`, or `undefined` when disabled. */
export function buildMinimizerOptions(group: ShellMinimizerSettings): MinimizerOptions | undefined {
	if (!group.enabled) return undefined;
	return {
		enabled: true,
		settingsPath: group.settingsPath || undefined,
		only: group.only.length > 0 ? group.only : undefined,
		except: group.except.length > 0 ? group.except : undefined,
		maxCaptureBytes: group.maxCaptureBytes,
	};
}

/**
 * Escape PowerShell `$env:VAR` references so brush-core's POSIX variable
 * expansion does not collapse them to `:VAR`.
 *
 * In POSIX a `:` terminates a parameter name, so `$env:SystemRoot` parses as
 * `${env}` followed by the literal `:SystemRoot`. `$env` is unset under brush,
 * which leaves the colon-prefixed remainder behind and mangles commands that
 * invoke PowerShell as a subprocess (e.g. `powershell -Command "Write-Host
 * $env:SystemRoot"`).
 *
 * The fix is purely textual: we walk the command, leave single-quoted regions
 * and existing `\X` escapes alone, and prepend `\` to every unescaped
 * `$env:<word-char>` occurrence outside single quotes. Brush then forwards
 * the literal `$env:NAME` to the PowerShell child where it has its native
 * meaning. The check is case-insensitive because PowerShell accepts `$Env:`,
 * `$ENV:` and `$env:` interchangeably.
 */
export function preservePowerShellEnvVars(command: string): string {
	let result = "";
	let inSingleQuote = false;
	let i = 0;
	while (i < command.length) {
		const ch = command[i];
		if (inSingleQuote) {
			if (ch === "'") inSingleQuote = false;
			result += ch;
			i++;
			continue;
		}
		if (ch === "'") {
			inSingleQuote = true;
			result += ch;
			i++;
			continue;
		}
		// Preserve existing backslash escapes verbatim so `\$env:` stays single-escaped.
		if (ch === "\\" && i + 1 < command.length) {
			result += command.slice(i, i + 2);
			i += 2;
			continue;
		}
		if (
			ch === "$" &&
			i + 5 < command.length &&
			command.slice(i + 1, i + 5).toLowerCase() === "env:" &&
			/[A-Za-z_]/.test(command[i + 5])
		) {
			result += "\\$";
			i++;
			continue;
		}
		result += ch;
		i++;
	}
	return result;
}

export async function executeBash(command: string, options?: BashExecutorOptions): Promise<BashResult> {
	const settings = await Settings.init();
	const { shell, env: shellEnv, prefix } = settings.getShellConfig();
	const snapshotPath = shell.includes("bash") ? await getOrCreateSnapshot(shell, shellEnv) : null;

	const minimizer = buildMinimizerOptions(settings.getGroup("shellMinimizer"));

	const commandCwd = await resolveShellCwd(options?.cwd);
	const commandEnv = options?.env ? { ...NON_INTERACTIVE_ENV, ...options.env } : NON_INTERACTIVE_ENV;

	// Apply command prefix if configured, then preserve any `$env:VAR` references
	// that PowerShell expects (brush would otherwise expand `$env` as an unset
	// POSIX parameter and leave the colon dangling — see #1079).
	const prefixedCommand = prefix ? `${prefix} ${command}` : command;
	const finalCommand = preservePowerShellEnvVars(prefixedCommand);

	// Create output sink for truncation and artifact handling
	const sink = new OutputSink({
		onChunk: options?.onChunk,
		artifactPath: options?.artifactPath,
		artifactId: options?.artifactId,
		headBytes: resolveOutputSinkHeadBytes(settings),
		maxColumns: resolveOutputMaxColumns(settings),
		// Throttle the streaming preview callback to avoid saturating the
		// event loop when commands produce massive output (e.g. seq 1 50M).
		chunkThrottleMs: options?.onChunk ? 50 : 0,
	});

	// sink.push() is synchronous — buffer management, counters, and onChunk
	// all run inline. File writes (artifact path) are handled asynchronously
	// inside the sink. No promise chain needed.
	const enqueueChunk = (chunk: string) => {
		sink.push(chunk);
	};

	if (options?.signal?.aborted) {
		return {
			exitCode: undefined,
			cancelled: true,
			...(await sink.dump("Command cancelled")),
		};
	}

	const sessionKey = buildSessionKey(shell, prefix, snapshotPath, shellEnv, options?.sessionKey, minimizer);
	const persistentSessionBroken = brokenShellSessions.has(sessionKey);
	if (persistentSessionBroken) {
		shellSessions.delete(sessionKey);
	}

	let shellSession = persistentSessionBroken ? undefined : shellSessions.get(sessionKey);
	if (!shellSession && !persistentSessionBroken) {
		shellSession = new Shell({
			sessionEnv: shellEnv,
			snapshotPath: snapshotPath ?? undefined,
			minimizer,
		});
		shellSessions.set(sessionKey, shellSession);
	}
	const userSignal = options?.signal;
	const runAbortController = new AbortController();
	const abortCurrentExecution = () => {
		if (!runAbortController.signal.aborted) {
			runAbortController.abort();
		}
		if (shellSession) {
			// Native abort is async; fire-and-forget because the caller races the command separately.
			void shellSession.abort();
		}
	};
	const abortHandler = () => {
		abortCurrentExecution();
	};
	if (userSignal) {
		userSignal.addEventListener("abort", abortHandler, { once: true });
	}

	let hardTimeoutTimer: NodeJS.Timeout | undefined;
	const hardTimeoutDeferred = Promise.withResolvers<"hard-timeout">();
	const baseTimeoutMs = Math.max(1_000, options?.timeout ?? 300_000);
	const hardTimeoutMs = baseTimeoutMs + HARD_TIMEOUT_GRACE_MS;
	hardTimeoutTimer = setTimeout(() => {
		abortCurrentExecution();
		hardTimeoutDeferred.resolve("hard-timeout");
	}, hardTimeoutMs);

	let resetSession = false;

	try {
		const runPromise = shellSession
			? shellSession.run(
					{
						command: finalCommand,
						cwd: commandCwd,
						env: commandEnv,
						timeoutMs: options?.timeout,
						signal: runAbortController.signal,
					},
					(err, chunk) => {
						if (!err) {
							enqueueChunk(chunk);
						}
					},
				)
			: executeShell(
					{
						command: finalCommand,
						cwd: commandCwd,
						env: commandEnv,
						sessionEnv: shellEnv,
						snapshotPath: snapshotPath ?? undefined,
						minimizer,
						timeoutMs: options?.timeout,
						signal: runAbortController.signal,
					},
					(err, chunk) => {
						if (!err) {
							enqueueChunk(chunk);
						}
					},
				);

		const winner = await Promise.race([
			runPromise.then(result => ({ kind: "result" as const, result })),
			hardTimeoutDeferred.promise.then(() => ({ kind: "hard-timeout" as const })),
		]);

		if (winner.kind === "hard-timeout") {
			if (shellSession) {
				resetSession = true;
				// Fall back to one-shot execution for the rest of the process once
				// a persistent session has stopped responding to cancellation.
				brokenShellSessions.add(sessionKey);
			}
			return {
				exitCode: undefined,
				cancelled: true,
				...(await sink.dump(`Command exceeded hard timeout after ${Math.round(hardTimeoutMs / 1000)} seconds`)),
			};
		}

		// Handle timeout
		if (winner.result.timedOut) {
			const annotation = options?.timeout
				? `Command timed out after ${Math.round(options.timeout / 1000)} seconds`
				: "Command timed out";
			resetSession = true;
			return {
				exitCode: undefined,
				cancelled: true,
				...(await sink.dump(annotation)),
			};
		}

		// Handle cancellation
		if (winner.result.cancelled) {
			resetSession = true;
			return {
				exitCode: undefined,
				cancelled: true,
				...(await sink.dump("Command cancelled")),
			};
		}

		// When the native minimizer rewrote the output, swap the sink's accumulated
		// raw stream for the minimized text, persist the original as a session
		// artifact, and splice an `artifact://<id>` footer into the visible text so
		// the agent can retrieve the raw bytes losslessly.
		const minimized = winner.result.minimized;
		if (minimized && minimized.text !== minimized.originalText) {
			sink.replace(minimized.text);
			if (options?.onMinimizedSave) {
				const artifactId = await options.onMinimizedSave(minimized.originalText, {
					filter: minimized.filter,
					inputBytes: minimized.inputBytes,
					outputBytes: minimized.outputBytes,
				});
				if (artifactId) {
					sink.push(`\n[raw output: artifact://${artifactId}]\n`);
				}
			}
		}

		// Normal completion
		return {
			exitCode: winner.result.exitCode,
			cancelled: false,
			...(await sink.dump()),
		};
	} catch (err) {
		resetSession = true;
		throw err;
	} finally {
		if (hardTimeoutTimer) {
			clearTimeout(hardTimeoutTimer);
		}
		if (userSignal) {
			userSignal.removeEventListener("abort", abortHandler);
		}
		if (resetSession) {
			shellSessions.delete(sessionKey);
		}
	}
}

function buildSessionKey(
	shell: string,
	prefix: string | undefined,
	snapshotPath: string | null,
	env: Record<string, string>,
	agentSessionKey?: string,
	minimizer?: MinimizerOptions,
): string {
	const entries = Object.entries(env);
	entries.sort(([a], [b]) => a.localeCompare(b));
	const envSerialized = entries.map(([key, value]) => `${key}=${value}`).join("\n");
	const minimizerSerialized = minimizer ? JSON.stringify(minimizer) : "";
	return [agentSessionKey ?? "", shell, prefix ?? "", snapshotPath ?? "", envSerialized, minimizerSerialized].join(
		"\n",
	);
}
