custom_css = """
    /* ============================================
       MAIN CONTAINER - Light theme
       ============================================ */
    .progress-text {
        display: none !important;
    }

    .gradio-container {
        max-width: 1000px !important;
        width: 100% !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background: #f8f9fa !important;
    }

    /* ============================================
       TABS
       ============================================ */
    button[role="tab"] {
        color: #6b7280 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        transition: all 0.2s ease !important;
        background: transparent !important;
    }

    button[role="tab"]:hover {
        color: #1f2937 !important;
    }

    button[role="tab"][aria-selected="true"] {
        color: #1f2937 !important;
        border-bottom: 2px solid #3b82f6 !important;
        border-radius: 0 !important;
        background: transparent !important;
        font-weight: 600 !important;
    }

    .tabs {
        border-bottom: none !important;
        border-radius: 0 !important;
    }

    .tab-nav {
        border-bottom: 1px solid #e5e7eb !important;
        border-radius: 0 !important;
    }

    button[role="tab"]::before,
    button[role="tab"]::after,
    .tabs::before,
    .tabs::after,
    .tab-nav::before,
    .tab-nav::after {
        display: none !important;
        content: none !important;
        border-radius: 0 !important;
    }

    #doc-management-tab {
        max-width: 500px !important;
        margin: 0 auto !important;
    }

    /* ============================================
       BUTTONS
       ============================================ */
    button {
        border-radius: 8px !important;
        border: none !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }

    .primary {
        background: #3b82f6 !important;
        color: white !important;
    }

    .primary:hover {
        background: #2563eb !important;
        transform: translateY(-1px) !important;
    }

    .stop {
        background: #ef4444 !important;
        color: white !important;
    }

    .stop:hover {
        background: #dc2626 !important;
        transform: translateY(-1px) !important;
    }

    /* ============================================
       CHAT INPUT BOX
       ============================================ */
    textarea[placeholder="Type a message..."],
    textarea[data-testid*="textbox"]:not(#file-list-box textarea) {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        box-shadow: none !important;
        color: #1f2937 !important;
    }

    textarea[placeholder="Type a message..."]:focus {
        background: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    .gr-text-input:has(textarea[placeholder="Type a message..."]),
    [class*="chatbot"] + * [data-testid="textbox"],
    form:has(textarea[placeholder="Type a message..."]) > div {
        background: transparent !important;
        border: none !important;
        gap: 12px !important;
    }

    form:has(textarea[placeholder="Type a message..."]) button,
    [class*="chatbot"] ~ * button[type="submit"] {
        background: transparent !important;
        border: none !important;
        padding: 8px !important;
    }

    form:has(textarea[placeholder="Type a message..."]) button:hover {
        background: rgba(59, 130, 246, 0.1) !important;
    }

    form:has(textarea[placeholder="Type a message..."]) {
        gap: 12px !important;
        display: flex !important;
    }

    /* ============================================
       FILE UPLOAD
       ============================================ */
    .file-preview,
    [data-testid="file-upload"] {
        background: #ffffff !important;
        border: 2px dashed #d1d5db !important;
        border-radius: 8px !important;
        color: #374151 !important;
        min-height: 200px !important;
    }

    .file-preview:hover,
    [data-testid="file-upload"]:hover {
        border-color: #3b82f6 !important;
        background: #f0f7ff !important;
    }

    .file-preview *,
    [data-testid="file-upload"] * {
        color: #374151 !important;
    }

    .file-preview .label,
    [data-testid="file-upload"] .label {
        display: none !important;
    }

    /* ============================================
       INPUTS & TEXTAREAS
       ============================================ */
    input,
    textarea {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        color: #1f2937 !important;
        transition: border-color 0.2s ease !important;
    }

    input:focus,
    textarea:focus {
        border-color: #3b82f6 !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    textarea[readonly] {
        background: #f3f4f6 !important;
        color: #4b5563 !important;
    }

    /* ============================================
       FILE LIST BOX
       ============================================ */
    #file-list-box {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    #file-list-box textarea {
        background: transparent !important;
        border: none !important;
        color: #374151 !important;
        padding: 0 !important;
    }

    /* ============================================
       CHATBOT
       ============================================ */
    .chatbot {
        border-radius: 8px !important;
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
    }

    .message {
        border-radius: 10px !important;
        width: fit-content !important;
    }

    .message.user {
        background: #3b82f6 !important;
        color: white !important;
    }

    .message.bot {
        background: #f3f4f6 !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
    }

    /* Force all chatbot text to be readable */
    .chatbot .message-wrap,
    .chatbot .message-wrap *,
    .chatbot .bot *,
    .chatbot p,
    .chatbot li,
    .chatbot span,
    .chatbot td,
    .chatbot th,
    .chatbot strong,
    .chatbot em,
    .chatbot h1, .chatbot h2, .chatbot h3, .chatbot h4 {
        color: #1f2937 !important;
    }

    .chatbot .user p,
    .chatbot .user li,
    .chatbot .user span {
        color: #ffffff !important;
    }

    /* Markdown code blocks in chat */
    .chatbot pre {
        background: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
    }

    .chatbot code {
        background: #f3f4f6 !important;
        color: #2563eb !important;
    }

    .chatbot hr {
        border-color: #e5e7eb !important;
    }

    /* ============================================
       PROGRESS BAR
       ============================================ */
    .progress-bar-wrap {
        border-radius: 10px !important;
        overflow: hidden !important;
        background: #e5e7eb !important;
    }

    .progress-bar {
        border-radius: 10px !important;
        background: #3b82f6 !important;
    }

    /* ============================================
       TYPOGRAPHY
       ============================================ */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
    }

    p, label, span {
        color: #374151 !important;
    }

    /* Markdown description text */
    .markdown-text, .prose, .gr-prose {
        color: #4b5563 !important;
    }

    /* ============================================
       GLOBAL OVERRIDES
       ============================================ */
    * {
        box-shadow: none !important;
    }

    footer {
        visibility: hidden;
    }
"""
