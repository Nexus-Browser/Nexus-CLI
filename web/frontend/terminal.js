/**
 * Terminal-specific functionality for Nexus CLI
 * Handles command completion, syntax highlighting, and terminal-specific features
 */

class TerminalEnhancer {
    constructor() {
        this.commands = [
            'help', 'generate', 'chat', 'analyze', 'file', 'create', 'edit', 'delete',
            'list', 'search', 'optimize', 'debug', 'test', 'deploy', 'status',
            'config', 'history', 'clear', 'exit', 'version', 'update'
        ];
        
        this.fileExtensions = [
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.css', '.html',
            '.json', '.xml', '.yml', '.yaml', '.md', '.txt', '.log', '.conf'
        ];
        
        this.syntaxPatterns = {
            keywords: /\b(function|class|const|let|var|if|else|for|while|return|import|export|from|async|await)\b/g,
            strings: /(["'`])((?:\\.|(?!\1)[^\\])*?)\1/g,
            comments: /(\/\/.*$|\/\*[\s\S]*?\*\/)/gm,
            numbers: /\b\d+\.?\d*\b/g,
            operators: /[+\-*/%=<>!&|]/g
        };
        
        this.init();
    }

    init() {
        this.setupCommandCompletion();
        this.setupSyntaxHighlighting();
        this.setupShortcuts();
        console.log('🔧 Terminal enhancer initialized');
    }

    setupCommandCompletion() {
        const input = document.getElementById('terminal-input');
        if (!input) return;

        let completionIndex = 0;
        let lastCompletion = '';
        let possibleCompletions = [];

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                
                const value = input.value;
                const cursorPos = input.selectionStart;
                const beforeCursor = value.substring(0, cursorPos);
                const afterCursor = value.substring(cursorPos);
                
                // Find the current word
                const words = beforeCursor.split(/\s+/);
                const currentWord = words[words.length - 1] || '';
                
                if (currentWord !== lastCompletion) {
                    // New completion request
                    possibleCompletions = this.getCompletions(currentWord, words.length === 1);
                    completionIndex = 0;
                } else {
                    // Cycle through completions
                    completionIndex = (completionIndex + 1) % possibleCompletions.length;
                }
                
                if (possibleCompletions.length > 0) {
                    const completion = possibleCompletions[completionIndex];
                    const newValue = beforeCursor.replace(new RegExp(this.escapeRegex(currentWord) + '$'), completion) + afterCursor;
                    
                    input.value = newValue;
                    input.setSelectionRange(
                        cursorPos - currentWord.length + completion.length,
                        cursorPos - currentWord.length + completion.length
                    );
                    
                    lastCompletion = completion;
                } else {
                    lastCompletion = '';
                }
            } else if (e.key !== 'Shift' && e.key !== 'Ctrl' && e.key !== 'Alt') {
                // Reset completion on other keys
                lastCompletion = '';
                completionIndex = 0;
            }
        });
    }

    getCompletions(partial, isFirstWord) {
        const completions = [];
        
        if (isFirstWord) {
            // Complete commands
            this.commands.forEach(cmd => {
                if (cmd.toLowerCase().startsWith(partial.toLowerCase())) {
                    completions.push(cmd);
                }
            });
        } else {
            // Complete file extensions or common words
            this.fileExtensions.forEach(ext => {
                if (ext.startsWith(partial) && partial.includes('.')) {
                    completions.push(partial + ext.substring(partial.length));
                }
            });
            
            // Add some common file/directory names
            const commonNames = ['src', 'test', 'docs', 'config', 'package.json', 'README.md'];
            commonNames.forEach(name => {
                if (name.toLowerCase().startsWith(partial.toLowerCase())) {
                    completions.push(name);
                }
            });
        }
        
        return completions.sort();
    }

    setupSyntaxHighlighting() {
        const outputContainer = document.getElementById('terminal-output');
        if (!outputContainer) return;

        // Observe for new content
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        this.highlightSyntax(node);
                    }
                });
            });
        });

        observer.observe(outputContainer, {
            childList: true,
            subtree: true
        });
    }

    highlightSyntax(element) {
        // Check if element contains code-like content
        const textContent = element.textContent || '';
        
        if (this.looksLikeCode(textContent)) {
            const highlighted = this.applySyntaxHighlighting(textContent);
            
            // Only apply if the element doesn't already have syntax highlighting
            if (!element.querySelector('.syntax-keyword')) {
                element.innerHTML = highlighted;
            }
        }
    }

    looksLikeCode(text) {
        // Simple heuristics to detect code
        const codeIndicators = [
            /function\s+\w+\s*\(/,
            /class\s+\w+/,
            /import\s+.*from/,
            /\w+\s*=\s*\w+/,
            /if\s*\(/,
            /for\s*\(/,
            /\w+\.\w+/,
            /\{[\s\S]*\}/,
            /\/\/|\/\*|\*\//
        ];
        
        return codeIndicators.some(pattern => pattern.test(text));
    }

    applySyntaxHighlighting(text) {
        let highlighted = this.escapeHtml(text);
        
        // Apply syntax highlighting patterns
        highlighted = highlighted.replace(this.syntaxPatterns.comments, '<span class="syntax-comment">$1</span>');
        highlighted = highlighted.replace(this.syntaxPatterns.strings, '<span class="syntax-string">$1$2$1</span>');
        highlighted = highlighted.replace(this.syntaxPatterns.keywords, '<span class="syntax-keyword">$1</span>');
        highlighted = highlighted.replace(this.syntaxPatterns.numbers, '<span class="syntax-number">$1</span>');
        highlighted = highlighted.replace(this.syntaxPatterns.operators, '<span class="syntax-operator">$1</span>');
        
        return highlighted;
    }

    setupShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+L - Clear terminal (like in real terminals)
            if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
                e.preventDefault();
                if (window.nexusCLI && window.nexusCLI.currentView === 'terminal') {
                    window.nexusCLI.clearTerminal();
                }
            }
            
            // Ctrl+C - Interrupt command (visual feedback)
            if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
                const terminalInput = document.getElementById('terminal-input');
                if (terminalInput && document.activeElement === terminalInput) {
                    e.preventDefault();
                    this.showInterruptMessage();
                }
            }
            
            // Ctrl+D - Show help
            if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
                e.preventDefault();
                if (window.nexusCLI && window.nexusCLI.currentView === 'terminal') {
                    this.showQuickHelp();
                }
            }
        });
    }

    showInterruptMessage() {
        const output = document.getElementById('terminal-output');
        const interruptMsg = document.createElement('div');
        interruptMsg.className = 'command-output';
        interruptMsg.innerHTML = '<span style="color: var(--warning-color);">^C</span>';
        output.appendChild(interruptMsg);
        
        // Clear the input
        const input = document.getElementById('terminal-input');
        input.value = '';
    }

    showQuickHelp() {
        const output = document.getElementById('terminal-output');
        const helpMsg = document.createElement('div');
        helpMsg.className = 'command-output';
        helpMsg.innerHTML = `
<div class="help-section">
  <strong>Quick Help - Nexus CLI Commands:</strong>
  
  <div class="help-commands">
    <div><code>help</code> - Show detailed help</div>
    <div><code>generate [description]</code> - Generate code</div>
    <div><code>chat [message]</code> - Chat with AI</div>
    <div><code>analyze [file]</code> - Analyze code</div>
    <div><code>create [filename]</code> - Create new file</div>
    <div><code>list</code> - List files</div>
    <div><code>clear</code> - Clear terminal</div>
  </div>
  
  <div class="help-shortcuts">
    <strong>Shortcuts:</strong>
    <div><kbd>Tab</kbd> - Command completion</div>
    <div><kbd>↑/↓</kbd> - Command history</div>
    <div><kbd>Ctrl+K</kbd> - Command palette</div>
    <div><kbd>Ctrl+L</kbd> - Clear terminal</div>
    <div><kbd>Ctrl+C</kbd> - Interrupt</div>
  </div>
</div>
        `;
        output.appendChild(helpMsg);
        
        // Scroll to bottom
        setTimeout(() => {
            output.scrollTop = output.scrollHeight;
        }, 10);
    }

    // Utility methods
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // Method to add custom commands
    addCommand(command) {
        if (!this.commands.includes(command)) {
            this.commands.push(command);
            this.commands.sort();
        }
    }

    // Method to add custom file extensions
    addFileExtension(extension) {
        if (!this.fileExtensions.includes(extension)) {
            this.fileExtensions.push(extension);
        }
    }
}

// Add CSS for syntax highlighting
const syntaxCSS = `
.syntax-keyword {
    color: #f97583;
    font-weight: 500;
}

.syntax-string {
    color: #85e89d;
}

.syntax-comment {
    color: var(--text-muted);
    font-style: italic;
}

.syntax-number {
    color: #79b8ff;
}

.syntax-operator {
    color: var(--primary-color);
}

.help-section {
    margin: 16px 0;
    color: var(--text-secondary);
}

.help-commands {
    margin: 12px 0;
    display: grid;
    gap: 4px;
}

.help-commands code {
    background: var(--bg-tertiary);
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--primary-color);
    font-family: 'Fira Code', monospace;
}

.help-shortcuts {
    margin-top: 12px;
    border-top: 1px solid var(--border-color);
    padding-top: 12px;
}

.help-shortcuts div {
    margin: 4px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.help-shortcuts kbd {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 11px;
    color: var(--text-muted);
}
`;

// Inject CSS
const style = document.createElement('style');
style.textContent = syntaxCSS;
document.head.appendChild(style);

// Initialize terminal enhancer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.terminalEnhancer = new TerminalEnhancer();
});
