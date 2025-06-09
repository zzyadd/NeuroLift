tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                primary: {
                    50: '#f0f9ff',
                    100: '#e0f2fe',
                    200: '#bae6fd',
                    300: '#7dd3fc',
                    400: '#38bdf8',
                    500: '#0ea5e9',
                    600: '#0284c7',
                    700: '#0369a1',
                    800: '#075985',
                    900: '#0c4a6e',
                }
            }
        }
    }
};

const supabaseClient = supabase.createClient(
    'https://xadjoggbirotiqenzxcb.supabase.co',
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhhZGpvZ2diaXJvdGlxZW56eGNiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MDc1NTI4MCwiZXhwIjoyMDU2MzMxMjgwfQ.zDSbRjMT-Y7Tmb2QaiKipfarOSy_PGvuGI132tLLVD4'
);

// Update API base URL to include a fallback
const API_BASE_URL = 'https://thrush-hot-radically.ngrok-free.app';  // Update this to match your local server

// Update API Integration Functions with better error handling
const API = {
    baseUrls: [
        'https://thrush-hot-radically.ngrok-free.app',  // Primary URL
    ],
    
    // Function to try multiple endpoints until one works
    async tryEndpoints(path, options) {
        let lastError = null;
        
        for (const baseUrl of this.baseUrls) {
            try {
                const response = await fetch(`${baseUrl}${path}`, options);
                
                if (response.ok) {
                    // If successful, update the primary URL to this working one
                    if (baseUrl !== this.baseUrls[0]) {
                        console.log(`Switched to working API endpoint: ${baseUrl}`);
                        // Optionally move this URL to the front of the array for future requests
                        this.baseUrls = [baseUrl, ...this.baseUrls.filter(url => url !== baseUrl)];
                    }
                    return response;
                }
                
                // If not ok but we got a response, capture the error message
                const errorData = await response.text();
                lastError = new Error(`Server responded with ${response.status}: ${errorData}`);
            } catch (error) {
                console.warn(`API endpoint ${baseUrl} failed:`, error);
                lastError = error;
                // Continue to next endpoint
            }
        }
        
        // If we get here, all endpoints failed
        throw lastError || new Error('All API endpoints failed');
    },

    async createChat(title, threadId) {
        try {
            const { data: { session } } = await supabaseClient.auth.getSession();
            if (!session) {
                throw new Error('No active session');
            }

            const options = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // 'Accept': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`,
                    "ngrok-skip-browser-warning": "69420"
                },
                body: JSON.stringify({
                    thread_id: threadId,
                    title: title
                })
            };

            const response = await this.tryEndpoints('/api/chats', options);
            return await response.json();
        } catch (error) {
            console.error('API Error in createChat:', error);
            throw new Error(`Failed to create chat: ${error.message}`);
        }
    },

    async sendMessage(message, threadId) {
        try {
            const { data: { session } } = await supabaseClient.auth.getSession();
            if (!session) {
                throw new Error('No active session');
            }
    
            const options = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`,
                    "ngrok-skip-browser-warning": "69420"
                },
                body: JSON.stringify({
                    prompt: message,
                    conversation_id: threadId,
                    stream: false,
                    use_memory: true
                })
            };
    
            // Log request for debugging
            console.log(`Sending request to chat-with-db for thread ${threadId}:`, message);
    
            const response = await this.tryEndpoints('/api/chat-with-db', options);
            
            // Try parsing the response
            try {
                const data = await response.json();
                console.log('Raw API response:', data);
                
                // Ensure the chat is marked as loaded since we're adding messages to it
                if (state.chats[threadId]) {
                    state.chats[threadId].isLoaded = true;
                }
                
                // Update the chat title in real-time if it has changed
                if (data.updated_title && state.currentChatId === threadId) {
                    state.chats[threadId].title = data.updated_title;
                    elements.currentChatTitle.textContent = data.updated_title;
                    updateChatHistoryList(); // Refresh the chat history list
                }
                
                return data;
            } catch (parseError) {
                console.error('Failed to parse JSON response:', parseError);
                
                // Try to get response as text if JSON parsing fails
                const textResponse = await response.text();
                console.log('Response as text:', textResponse);
                
                return { response: textResponse };
            }
        } catch (error) {
            console.error('API Error in sendMessage:', error);
            throw error;
        }
    },

    async getUserChats() {
        try {
            const { data: { session } } = await supabaseClient.auth.getSession();
            if (!session) {
                throw new Error('No active session');
            }

            const options = {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    "ngrok-skip-browser-warning": "69420"
                }
            };

            const response = await this.tryEndpoints('/api/user-chats', options);
            return await response.json();
        } catch (error) {
            console.error('API Error in getUserChats:', error);
            throw new Error(`Failed to load chats: ${error.message}`);
        }
    },

    async getConversationHistory(threadId) {
        try {
            const { data: { session } } = await supabaseClient.auth.getSession();
            if (!session) {
                throw new Error('No active session');
            }

            const options = {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    "ngrok-skip-browser-warning": "69420"
                }
            };

            // First get chat by thread_id
            const chatResponse = await this.tryEndpoints(`/api/chats/thread/${threadId}`, options);
            const chat = await chatResponse.json();

            // Then get conversations using chat.id
            const convResponse = await this.tryEndpoints(`/api/chats/${chat.id}/conversations`, options);
            const conversations = await convResponse.json();

            // Include the title in the returned data
            return conversations.map(conv => ({
                ...conv,
                title: chat.title
            }));
        } catch (error) {
            console.error('API Error in getConversationHistory:', error);
            throw new Error(`Failed to load conversation history: ${error.message}`);
        }
    }
};
