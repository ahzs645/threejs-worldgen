/** A fatal backend failure survives asynchronous loading, progress updates and state replacement. */
export class EngineFailure {
  message: string | null = null
  get failed(): boolean {
    return this.message !== null
  }
  report(message: string, notify: (message: string) => void): boolean {
    if (this.failed) return false
    this.message = message
    notify(message)
    return true
  }
  check(): void {
    if (this.message !== null) throw new Error(this.message)
  }
  progress(message: string): string {
    return this.message ?? message
  }
  decorate(state: Record<string, unknown>): Record<string, unknown> {
    return this.message === null ? state : { ...state, ready: true, error: this.message, lastError: this.message }
  }
}
