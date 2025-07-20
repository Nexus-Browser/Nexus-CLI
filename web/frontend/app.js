/**
 * Nexus CLI Web Terminal - Main Application
 * Warp-inspired terminal interface with iLLuMinator-4.7B AI
 */

class NexusCLIApp {
    constructor() {
        this.socket = null;
        this.sessionId = null;
        this.currentView = 'terminal';
        this.commandHistory = [];
        this.historyIndex = -1;
        this.sessionStartTime = Date.now();
        
        this.init();
    }

    async init() {
        await this.initializeSession();
        this.initializeWebSocket();
        this.bindEvents();
        this.startSessionTimer();
        this.loadCommandHistory();
        
        console.log('🚀 Nexus CLI Web Terminal initialized');
    }

    async initializeSession() {
        try {
            const response = await fetch('/api/session/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.sessionId = data.session_id;
                document.getElementById('session-id').textContent = `Session: ${this.sessionId.slice(0, 8)}`;
                console.log('✅ Session created:', this.sessionId);
            } else {
                console.error('❌ Failed to create session');
                this.sessionId = 'local-' + Date.now();
            }
        } catch (error) {
            console.error('❌ Session initialization error:', error);
            this.sessionId = 'local-' + Date.now();
        }
    }

    initializeWebSocket() {
        if (!this.sessionId) return;
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
        
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
            console.log('🔌 WebSocket connected');
            this.updateConnectionStatus(true);
        };
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.socket.onclose = () => {
            console.log('🔌 WebSocket disconnected');
            this.updateConnectionStatus(false);
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => {
                if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
                    this.initializeWebSocket();
                }
            }, 3000);
        };
        
        this.socket.onerror = (error) => {
            console.error('🔌 WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'command_result':
                this.displayCommandResult(data.result);
                break;
            case 'chat_response':
                this.displayChatMessage(data.message, 'bot');
                break;
            case 'code_generated':
                this.displayGeneratedCode(data.code, data.language);
                break;
            case 'status_update':
                this.updateStatus(data.status);
                break;
            default:
                console.log('📨 Unknown message type:', data.type);
        }
    }

    bindEvents() {
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.currentTarget.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Command palette
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.toggleCommandPalette();
            }
            
            if (e.key === 'Escape') {
                this.hideCommandPalette();
            }
        });

        // Terminal input
        const terminalInput = document.getElementById('terminal-input');
        terminalInput.addEventListener('keydown', (e) => {
            this.handleTerminalKeydown(e);
        });

        // Chat input
        const chatInput = document.getElementById('chat-input');
        const sendChatBtn = document.getElementById('send-chat');
        
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendChatMessage();
            }
        });
        
        sendChatBtn.addEventListener('click', () => {
            this.sendChatMessage();
        });

        // Auto-resize chat textarea
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        });

        // Code generation
        document.getElementById('generate-code').addEventListener('click', () => {
            this.generateCode();
        });

        // Code generation input
        document.getElementById('code-prompt').addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.generateCode();
            }
        });

        // Control buttons
        document.getElementById('clear-terminal').addEventListener('click', () => {
            this.clearTerminal();
        });
        
        document.getElementById('clear-chat').addEventListener('click', () => {
            this.clearChat();
        });

        document.getElementById('copy-code').addEventListener('click', () => {
            this.copyGeneratedCode();
        });

        document.getElementById('fullscreen-btn').addEventListener('click', () => {
            this.toggleFullscreen();
        });

        // Settings button
        document.getElementById('settings-btn').addEventListener('click', () => {
            this.openSettings();
        });

        // Split terminal button
        document.getElementById('split-terminal').addEventListener('click', () => {
            this.splitTerminal();
        });

        // New tab button
        document.getElementById('new-tab-btn').addEventListener('click', () => {
            this.createNewTab();
        });

        // Save code button
        document.getElementById('save-code').addEventListener('click', () => {
            this.saveGeneratedCode();
        });

        // Tab close buttons
        document.querySelectorAll('.tab-close').forEach(closeBtn => {
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.closeTab(e.target.closest('.tab'));
            });
        });

        // Command suggestions
        document.querySelectorAll('.suggestion').forEach(suggestion => {
            suggestion.addEventListener('click', (e) => {
                const command = e.currentTarget.dataset.command;
                this.executeQuickCommand(command);
            });
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });

        // Update views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.toggle('active', view.id === `${tabName}-view`);
        });

        this.currentView = tabName;
        
        // Focus appropriate input
        setTimeout(() => {
            if (tabName === 'terminal') {
                document.getElementById('terminal-input').focus();
            } else if (tabName === 'chat') {
                document.getElementById('chat-input').focus();
            } else if (tabName === 'code') {
                document.getElementById('code-prompt').focus();
            }
        }, 100);
    }

    toggleCommandPalette() {
        const palette = document.getElementById('command-palette');
        const paletteInput = document.getElementById('palette-input');
        
        palette.classList.toggle('active');
        
        if (palette.classList.contains('active')) {
            setTimeout(() => paletteInput.focus(), 100);
        }
    }

    hideCommandPalette() {
        document.getElementById('command-palette').classList.remove('active');
    }

    executeQuickCommand(command) {
        this.hideCommandPalette();
        
        switch (command) {
            case 'generate':
                this.switchTab('code');
                break;
            case 'chat':
                this.switchTab('chat');
                break;
            case 'analyze':
                this.switchTab('terminal');
                document.getElementById('terminal-input').value = 'analyze ';
                document.getElementById('terminal-input').focus();
                break;
            default:
                console.log('Unknown quick command:', command);
        }
    }

    async handleTerminalKeydown(e) {
        const input = e.target;
        
        if (e.key === 'Enter') {
            e.preventDefault();
            const command = input.value.trim();
            
            if (command) {
                this.executeCommand(command);
                this.commandHistory.unshift(command);
                this.historyIndex = -1;
                input.value = '';
            }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (this.historyIndex < this.commandHistory.length - 1) {
                this.historyIndex++;
                input.value = this.commandHistory[this.historyIndex] || '';
            }
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (this.historyIndex > -1) {
                this.historyIndex--;
                input.value = this.historyIndex >= 0 ? this.commandHistory[this.historyIndex] : '';
            }
        } else if (e.key === 'Tab') {
            e.preventDefault();
            // TODO: Implement command completion
        }
    }

    async executeCommand(command) {
        this.displayCommand(command);
        this.showLoading();

        try {
            const response = await fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    command: command,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.displayCommandResult(data.result);
            } else {
                this.displayCommandResult(`Error: ${data.detail || 'Command failed'}`, 'error');
            }
        } catch (error) {
            console.error('Command execution error:', error);
            this.displayCommandResult(`Error: Failed to execute command`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async executeTerminalCommand(command, outputElement) {
        // Display the command in the specified output element
        const commandLine = document.createElement('div');
        commandLine.className = 'command-line';
        commandLine.innerHTML = `
            <span class="terminal-prompt">nexus@web:~$</span>
            <span class="command-text">${this.escapeHtml(command)}</span>
        `;
        outputElement.appendChild(commandLine);

        try {
            const response = await fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    command: command,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                const resultLine = document.createElement('div');
                resultLine.className = 'command-result';
                resultLine.innerHTML = `<pre>${this.escapeHtml(data.output || 'Command executed successfully')}</pre>`;
                outputElement.appendChild(resultLine);
            } else {
                const errorLine = document.createElement('div');
                errorLine.className = 'command-result error';
                errorLine.innerHTML = `<pre>Error: ${this.escapeHtml(data.detail || 'Command failed')}</pre>`;
                outputElement.appendChild(errorLine);
            }
        } catch (error) {
            const errorLine = document.createElement('div');
            errorLine.className = 'command-result error';
            errorLine.innerHTML = `<pre>Error: Failed to execute command</pre>`;
            outputElement.appendChild(errorLine);
        }
        
        // Scroll to bottom
        this.scrollToBottom(outputElement);
    }

    displayCommand(command) {
        const output = document.getElementById('terminal-output');
        const commandLine = document.createElement('div');
        commandLine.className = 'command-line';
        commandLine.innerHTML = `
            <span class="prompt">
                <span class="prompt-user">nexus</span>
                <span class="prompt-separator">@</span>
                <span class="prompt-host">cli</span>
                <span class="prompt-path">~/</span>
                <span class="prompt-symbol">$</span>
            </span>
            <span class="command-text">${this.escapeHtml(command)}</span>
        `;
        
        output.appendChild(commandLine);
        this.scrollToBottom(output);
    }

    displayCommandResult(result, type = 'normal') {
        const output = document.getElementById('terminal-output');
        const resultDiv = document.createElement('div');
        resultDiv.className = `command-output ${type}`;
        resultDiv.textContent = result;
        
        output.appendChild(resultDiv);
        this.scrollToBottom(output);
    }

    async sendChatMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        this.displayChatMessage(message, 'user');
        input.value = '';
        input.style.height = 'auto';
        
        this.showLoading();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.displayChatMessage(data.response, 'bot');
            } else {
                this.displayChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.displayChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
        } finally {
            this.hideLoading();
        }
    }

    displayChatMessage(message, sender) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const timestamp = new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        const avatar = sender === 'bot' ? 
            '<i class="fas fa-robot"></i>' : 
            '<i class="fas fa-user"></i>';
        
        const senderName = sender === 'bot' ? 'iLLuMinator-4.7B' : 'You';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender">${senderName}</span>
                    <span class="timestamp">${timestamp}</span>
                </div>
                <div class="message-text">${this.formatChatMessage(message)}</div>
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom(messagesContainer);
    }

    async generateCode() {
        const prompt = document.getElementById('code-prompt').value.trim();
        const language = document.getElementById('language-select').value;
        
        if (!prompt) {
            alert('Please enter a description of what you want to build.');
            return;
        }
        
        this.showLoading();

        try {
            const response = await fetch('/api/code/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    language: language,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.displayGeneratedCode(data.code, language);
            } else {
                alert('Failed to generate code. Please try again.');
            }
        } catch (error) {
            console.error('Code generation error:', error);
            alert('Failed to generate code. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayGeneratedCode(code, language) {
        const outputElement = document.getElementById('code-output');
        const codeElement = outputElement.querySelector('code');
        
        codeElement.textContent = code;
        codeElement.className = `language-${language}`;
        
        // Trigger syntax highlighting
        if (typeof Prism !== 'undefined') {
            Prism.highlightElement(codeElement);
        }
    }

    copyGeneratedCode() {
        const codeElement = document.querySelector('#code-output code');
        const code = codeElement.textContent;
        
        navigator.clipboard.writeText(code).then(() => {
            // Visual feedback
            const copyBtn = document.getElementById('copy-code');
            const originalIcon = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
            
            setTimeout(() => {
                copyBtn.innerHTML = originalIcon;
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy code:', err);
        });
    }

    clearTerminal() {
        const output = document.getElementById('terminal-output');
        output.innerHTML = `
            <div class="welcome-message">
                <div class="ascii-art">
    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗     ██████╗██╗     ██╗
    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ██╔════╝██║     ██║
    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗    ██║     ██║     ██║
    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║    ██║     ██║     ██║
    ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║    ╚██████╗███████╗██║
    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝     ╚═════╝╚══════╝╚═╝
                </div>
                <div class="welcome-text">
                    <p>Welcome to <strong>Nexus CLI Web Terminal</strong></p>
                    <p>Powered by <strong>iLLuMinator-4.7B</strong> AI Model</p>
                    <p>Type <code>help</code> to get started or use <kbd>Ctrl+K</kbd> for command palette</p>
                </div>
            </div>
        `;
    }

    clearChat() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="sender">iLLuMinator-4.7B</span>
                        <span class="timestamp">Now</span>
                    </div>
                    <div class="message-text">
                        Hello! I'm your AI coding assistant. I can help you with:
                        <ul>
                            <li>Code generation and debugging</li>
                            <li>Architecture advice and best practices</li>
                            <li>Code analysis and optimization</li>
                            <li>Technical questions and explanations</li>
                        </ul>
                        What would you like to work on today?
                    </div>
                </div>
            </div>
        `;
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
            document.getElementById('fullscreen-btn').innerHTML = '<i class="fas fa-compress"></i>';
        } else {
            document.exitFullscreen();
            document.getElementById('fullscreen-btn').innerHTML = '<i class="fas fa-expand"></i>';
        }
    }

    showLoading() {
        document.getElementById('loading-overlay').classList.add('active');
    }

    hideLoading() {
        document.getElementById('loading-overlay').classList.remove('active');
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        const statusDot = statusElement.querySelector('.status-dot');
        const statusText = statusElement.querySelector('span');
        
        if (connected) {
            statusDot.style.background = 'var(--success-color)';
            statusText.textContent = 'Connected';
        } else {
            statusDot.style.background = 'var(--error-color)';
            statusText.textContent = 'Disconnected';
        }
    }

    updateStatus(status) {
        // Update various status indicators based on the status object
        if (status.memory_usage) {
            document.getElementById('memory-usage').textContent = `Memory: ${status.memory_usage}`;
        }
    }

    startSessionTimer() {
        setInterval(() => {
            const elapsed = Date.now() - this.sessionStartTime;
            const seconds = Math.floor(elapsed / 1000) % 60;
            const minutes = Math.floor(elapsed / (1000 * 60)) % 60;
            const hours = Math.floor(elapsed / (1000 * 60 * 60));
            
            const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            document.getElementById('session-time').textContent = timeString;
        }, 1000);
    }

    loadCommandHistory() {
        const saved = localStorage.getItem('nexus-cli-history');
        if (saved) {
            try {
                this.commandHistory = JSON.parse(saved);
            } catch (error) {
                console.error('Failed to load command history:', error);
            }
        }
    }

    saveCommandHistory() {
        try {
            localStorage.setItem('nexus-cli-history', JSON.stringify(this.commandHistory.slice(0, 100)));
        } catch (error) {
            console.error('Failed to save command history:', error);
        }
    }

    formatChatMessage(message) {
        // Basic markdown-like formatting
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom(element) {
        setTimeout(() => {
            element.scrollTop = element.scrollHeight;
        }, 10);
    }

    // New button implementations
    openSettings() {
        // Create settings modal
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h3>Settings</h3>
                    <button class="btn-icon" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="setting-group">
                        <label>Theme</label>
                        <select id="theme-select">
                            <option value="dark">Dark</option>
                            <option value="light">Light</option>
                            <option value="auto">Auto</option>
                        </select>
                    </div>
                    <div class="setting-group">
                        <label>Font Size</label>
                        <input type="range" id="font-size" min="12" max="20" value="14">
                        <span id="font-size-value">14px</span>
                    </div>
                    <div class="setting-group">
                        <label>AI Model Temperature</label>
                        <input type="range" id="temperature" min="0.1" max="1.0" step="0.1" value="0.7">
                        <span id="temperature-value">0.7</span>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-primary" onclick="nexusCLI.saveSettings(); this.closest('.modal-overlay').remove();">
                        Save Settings
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Add event listeners for real-time updates
        modal.querySelector('#font-size').addEventListener('input', (e) => {
            const value = e.target.value + 'px';
            document.getElementById('font-size-value').textContent = value;
            document.documentElement.style.setProperty('--font-size', value);
        });
        
        modal.querySelector('#temperature').addEventListener('input', (e) => {
            document.getElementById('temperature-value').textContent = e.target.value;
        });
    }

    saveSettings() {
        const settings = {
            theme: document.getElementById('theme-select').value,
            fontSize: document.getElementById('font-size').value,
            temperature: document.getElementById('temperature').value
        };
        localStorage.setItem('nexus-cli-settings', JSON.stringify(settings));
        console.log('Settings saved:', settings);
    }

    splitTerminal() {
        const terminalContainer = document.querySelector('.terminal-container');
        const existingSplits = terminalContainer.querySelectorAll('.terminal-split').length;
        
        if (existingSplits >= 3) {
            this.showNotification('Maximum of 4 terminal splits reached');
            return;
        }

        const newSplit = document.createElement('div');
        newSplit.className = 'terminal-split';
        newSplit.innerHTML = `
            <div class="terminal-header">
                <span class="terminal-title">Terminal ${existingSplits + 2}</span>
                <button class="btn-close" onclick="this.closest('.terminal-split').remove()">×</button>
            </div>
            <div class="terminal-content">
                <div class="terminal-output"></div>
                <div class="terminal-input-line">
                    <span class="terminal-prompt">nexus@web:~$</span>
                    <input type="text" class="terminal-input" placeholder="Enter command...">
                </div>
            </div>
        `;
        
        terminalContainer.appendChild(newSplit);
        
        // Add event listener for the new terminal input
        const newInput = newSplit.querySelector('.terminal-input');
        newInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const command = e.target.value.trim();
                if (command) {
                    this.executeTerminalCommand(command, newSplit.querySelector('.terminal-output'));
                    e.target.value = '';
                }
            }
        });
        
        newInput.focus();
    }

    createNewTab() {
        const tabsContainer = document.querySelector('.tabs');
        const tabCount = tabsContainer.querySelectorAll('.tab:not(.new-tab-btn)').length;
        
        const newTab = document.createElement('div');
        newTab.className = 'tab';
        newTab.dataset.tab = `tab-${tabCount + 1}`;
        newTab.innerHTML = `
            <span class="tab-title">Terminal ${tabCount + 1}</span>
            <button class="tab-close">×</button>
        `;
        
        // Insert before the new tab button
        const newTabBtn = document.querySelector('.new-tab-btn');
        tabsContainer.insertBefore(newTab, newTabBtn);
        
        // Add click handlers
        newTab.addEventListener('click', (e) => {
            if (!e.target.classList.contains('tab-close')) {
                this.switchTab(newTab.dataset.tab);
            }
        });
        
        newTab.querySelector('.tab-close').addEventListener('click', (e) => {
            e.stopPropagation();
            this.closeTab(newTab);
        });
        
        this.switchTab(newTab.dataset.tab);
    }

    closeTab(tab) {
        const tabsContainer = tab.parentNode;
        const remainingTabs = tabsContainer.querySelectorAll('.tab:not(.new-tab-btn)');
        
        if (remainingTabs.length <= 1) {
            this.showNotification('Cannot close the last tab');
            return;
        }
        
        const wasActive = tab.classList.contains('active');
        tab.remove();
        
        if (wasActive) {
            const firstTab = tabsContainer.querySelector('.tab:not(.new-tab-btn)');
            if (firstTab) {
                this.switchTab(firstTab.dataset.tab);
            }
        }
    }

    saveGeneratedCode() {
        const codeOutput = document.getElementById('code-output');
        const code = codeOutput.textContent || codeOutput.innerText;
        
        if (!code || code.trim() === '') {
            this.showNotification('No code to save');
            return;
        }
        
        const language = document.getElementById('code-language').value || 'txt';
        const filename = prompt(`Enter filename (without extension):`, `generated_code`);
        
        if (filename) {
            const blob = new Blob([code], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${filename}.${language === 'javascript' ? 'js' : language}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showNotification(`Code saved as ${a.download}`);
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.nexusCLI = new NexusCLIApp();
});

// Save command history on page unload
window.addEventListener('beforeunload', () => {
    if (window.nexusCLI) {
        window.nexusCLI.saveCommandHistory();
    }
});

// Handle fullscreen change
document.addEventListener('fullscreenchange', () => {
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    if (document.fullscreenElement) {
        fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i>';
    } else {
        fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
    }
});
