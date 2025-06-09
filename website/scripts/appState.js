// appState.js
const state = {
    currentUser: null,
    chats: {},
    currentChatId: null,
    isAdmin: false,
    darkMode: localStorage.getItem('darkMode') === 'true'
};

// DOM Elements
const elements = {
    authModal: document.getElementById('authModal'),
    appContainer: document.getElementById('appContainer'),
    loginForm: document.getElementById('loginForm'),
    signupForm: document.getElementById('signupForm'),
    authTitle: document.getElementById('authTitle'),
    sidebar: document.getElementById('sidebar'),
    openSidebar: document.getElementById('openSidebar'),
    closeSidebar: document.getElementById('closeSidebar'),
    chatHistory: document.getElementById('chatHistory'),
    newChatBtn: document.getElementById('newChatBtn'),
    chatContainer: document.getElementById('chatContainer'),
    chatInput: document.getElementById('chatInput'),
    sendMessageBtn: document.getElementById('sendMessageBtn'),
    currentChatTitle: document.getElementById('currentChatTitle'),
    deleteChatBtn: document.getElementById('deleteChatBtn'),
    saveChatBtn: document.getElementById('saveChatBtn'),
    accountSettingsBtn: document.getElementById('accountSettingsBtn'),
    accountSettingsModal: document.getElementById('accountSettingsModal'),
    darkModeToggle: document.getElementById('darkModeToggle'),
    logoutBtn: document.getElementById('logoutBtn'),
    userDashboard: document.getElementById('userDashboard'),
    adminDashboard: document.getElementById('adminDashboard'),
    adminDashboardBtn: document.getElementById('adminDashboardBtn'),
    backToChatBtn: document.getElementById('backToChatBtn'),
    usernameDisplay: document.getElementById('usernameDisplay')
};

// Initialize the app
async function init() {
    // Check for existing session
    const { data: { session } } = await supabaseClient.auth.getSession();

    if (session) {
        state.currentUser = {
            name: session.user.user_metadata.name || session.user.email,
            email: session.user.email,
            id: session.user.id,
            session: session
        };
        showApp();
        await loadUserChats();
    } else {
        showAuthModal();
    }

    // Set dark mode if enabled
    if (state.darkMode) {
        document.documentElement.classList.add('dark');
    }

    // Event listeners
    elements.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    elements.sendMessageBtn.addEventListener('click', sendMessage);
    elements.newChatBtn.addEventListener('click', createNewChat);
    elements.deleteChatBtn.addEventListener('click', deleteCurrentChat);
    elements.saveChatBtn.addEventListener('click', saveCurrentChat);
    elements.accountSettingsBtn.addEventListener('click', () => {
        document.getElementById('accountName').value = state.currentUser?.name || '';
        document.getElementById('accountEmail').value = state.currentUser?.email || '';
        elements.accountSettingsModal.classList.remove('hidden');
    });
    elements.darkModeToggle.addEventListener('click', toggleDarkMode);
    elements.logoutBtn.addEventListener('click', logout);
    elements.adminDashboardBtn.addEventListener('click', showAdminDashboard);
    elements.backToChatBtn.addEventListener('click', showUserDashboard);
    elements.openSidebar.addEventListener('click', () => elements.sidebar.classList.remove('-translate-x-full'));
    elements.closeSidebar.addEventListener('click', () => elements.sidebar.classList.add('-translate-x-full'));

}

// Auth Functions
function toggleAuthForm(formType) {
    if (formType === 'login') {
        elements.loginForm.classList.remove('hidden');
        elements.signupForm.classList.add('hidden');
        elements.authTitle.textContent = 'Welcome Back';
    } else {
        elements.loginForm.classList.add('hidden');
        elements.signupForm.classList.remove('hidden');
        elements.authTitle.textContent = 'Create Account';
    }
}

async function login() {
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const rememberMe = document.getElementById('rememberMe').checked;

    try {
        // Regular user login
        const { data: { user, session }, error } = await supabaseClient.auth.signInWithPassword({
            email,
            password
        });

        if (error) throw error;

        // Check if user has admin role in app_metadata
        const isAdmin = user.app_metadata && 
                       (user.app_metadata.role === 'admin' || 
                        user.app_metadata.is_super_admin === true);

        state.currentUser = {
            name: user.user_metadata.name || user.email,
            email: user.email,
            id: user.id,
            session: session
        };
        state.isAdmin = isAdmin; // Set admin flag based on metadata

        if (rememberMe) {
            localStorage.setItem('currentUser', JSON.stringify(state.currentUser));
        }

        showApp();
        await loadUserChats(); // Load user's chats after login
    } catch (error) {
        alert('Error logging in: ' + error.message);
    }
}

async function signup() {
    const name = document.getElementById('signupName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('signupConfirmPassword').value;

    // Basic validation
    if (!name || !email || !password) {
        alert('Please fill in all required fields');
        return;
    }

    if (password !== confirmPassword) {
        alert('Passwords do not match');
        return;
    }

    try {
        // Register user with Supabase
        const { data: { user }, error } = await supabaseClient.auth.signUp({
            email,
            password,
            options: {
                data: {
                    name: name
                }
            }
        });

        if (error) throw error;

        // Check if email confirmation is required
        if (user && !user.confirmed_at) {
            alert('Please check your email to confirm your account before logging in.');
            toggleAuthForm('login');
            return;
        }

        // Auto login after signup if email confirmation is not required
        if (user) {
            state.currentUser = {
                name: name,
                email: user.email,
                id: user.id,
                session: user.session
            };
            state.isAdmin = false;
            showApp();
            createNewChat(); // Create initial chat for new user
        }
    } catch (error) {
        console.error('Error signing up:', error);
        alert('Error signing up: ' + error.message);
    }
}

async function logout() {
    try {
        await supabaseClient.auth.signOut();
        state.currentUser = null;
        state.chats = {};
        state.currentChatId = null;
        state.isAdmin = false;
        localStorage.removeItem('currentUser');
        showAuthModal();
    } catch (error) {
        console.error('Error logging out:', error);
        alert('Error logging out. Please try again.');
    }
}

// UI Functions
function showAuthModal() {
    elements.authModal.classList.remove('hidden');
    elements.appContainer.classList.add('hidden');
    document.getElementById('loginEmail').value = '';
    document.getElementById('loginPassword').value = '';
    document.getElementById('signupName').value = '';
    document.getElementById('signupEmail').value = '';
    document.getElementById('signupPassword').value = '';
    document.getElementById('signupConfirmPassword').value = '';
    toggleAuthForm('login');
}

function showApp() {
    elements.authModal.classList.add('hidden');
    elements.appContainer.classList.remove('hidden');
    elements.usernameDisplay.textContent = state.currentUser.name;

    // Show admin button only for admin
    if (state.isAdmin) {
        elements.adminDashboardBtn.classList.remove('hidden');
    } else {
        elements.adminDashboardBtn.classList.add('hidden');
    }

    // Load or create initial chat
    if (Object.keys(state.chats).length === 0) {
        createNewChat();
    } else {
        loadChat(state.currentChatId);
    }
}

function toggleDarkMode() {
    state.darkMode = !state.darkMode;
    localStorage.setItem('darkMode', state.darkMode);
    document.documentElement.classList.toggle('dark', state.darkMode);
}

function showUserDashboard() {
    elements.userDashboard.classList.remove('hidden');
    elements.adminDashboard.classList.add('hidden');
}

function showAdminDashboard() {
    elements.userDashboard.classList.add('hidden');
    elements.adminDashboard.classList.remove('hidden');
    renderAdminDashboard();
}

// Chat Functions
async function createNewChat() {
    try {
        const timestamp = Date.now();
        const threadId = `thread_${timestamp}`;
        const chatData = await API.createChat('New Chat', threadId);

        state.currentChatId = threadId; // Store thread_id as reference
        state.chats[threadId] = {
            id: threadId,
            dbId: chatData.id, // Store the database ID
            title: 'New Chat',
            messages: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        updateChatHistoryList();
        loadChat(threadId);
    } catch (error) {
        console.error('Error creating chat:', error);
        alert('Error creating chat: ' + error.message);
    }
}

async function loadUserChats() {
    try {
        const chats = await API.getUserChats();
        state.chats = {};

        for (const chat of chats) {
            // Store chat metadata without loading conversation history
            state.chats[chat.thread_id] = {
                id: chat.thread_id,
                dbId: chat.id,
                title: chat.title,
                messages: [], // Initialize with empty messages array
                createdAt: chat.created_at,
                updatedAt: chat.created_at,
                isLoaded: false // Flag to track if conversation has been loaded
            };
        }

        updateChatHistoryList();
        
        // If there are chats, select the first one but don't load messages yet
        if (Object.keys(state.chats).length > 0) {
            const firstChatId = Object.keys(state.chats)[0];
            state.currentChatId = firstChatId;
            elements.currentChatTitle.textContent = state.chats[firstChatId].title;
            
            // Show empty chat container with loading message
            elements.chatContainer.innerHTML = `
                <div class="text-center py-10 text-gray-500 dark:text-gray-400">
                    <i class="fas fa-robot text-4xl mb-2"></i>
                    <p class="text-lg">How can I help you today?</p>
                </div>
            `;
            
            // Highlight the selected chat
            document.querySelectorAll('#chatHistory li').forEach(item => {
                item.classList.toggle('bg-blue-50', item.dataset.chatId === firstChatId);
                item.classList.toggle('dark:bg-gray-700', item.dataset.chatId === firstChatId);
            });
        }
    } catch (error) {
        console.error('Error loading chats:', error);
    }
}

async function loadChat(chatId) {
    if (!state.chats[chatId]) return;

    state.currentChatId = chatId;
    const chat = state.chats[chatId];
    
    // Update UI title immediately
    elements.currentChatTitle.textContent = chat.title;
    
    // Show loading indicator
    elements.chatContainer.innerHTML = `
        <div class="flex items-center justify-center h-full">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
    `;
    
    // Only load conversation history if not already loaded
    if (!chat.isLoaded && chat.id) {
        try {
            const history = await API.getConversationHistory(chat.id);
            state.chats[chatId].messages = history.map(msg => ({
                role: msg.type,
                content: msg.content,
                timestamp: msg.created_at
            }));
            state.chats[chatId].isLoaded = true; // Mark as loaded
        } catch (error) {
            console.error('Error loading conversation history:', error);
            elements.chatContainer.innerHTML = `
                <div class="text-center py-10 text-red-500">
                    <i class="fas fa-exclamation-triangle text-4xl mb-2"></i>
                    <p class="text-lg">Failed to load conversation. Please try again.</p>
                </div>
            `;
            return;
        }
    }

    // Update UI with messages
    elements.chatContainer.innerHTML = '';

    if (chat.messages.length === 0) {
        elements.chatContainer.innerHTML = `
            <div class="text-center py-10 text-gray-500 dark:text-gray-400">
                <i class="fas fa-robot text-4xl mb-2"></i>
                <p class="text-lg">How can I help you today?</p>
            </div>
        `;
    } else {
        chat.messages.forEach(message => {
            addMessageToChat(message.role, message.content);
        });
    }

    // Highlight current chat in history
    document.querySelectorAll('#chatHistory li').forEach(item => {
        item.classList.toggle('bg-blue-50', item.dataset.chatId === chatId);
        item.classList.toggle('dark:bg-gray-700', item.dataset.chatId === chatId);
    });
}

function deleteCurrentChat() {
    if (!state.currentChatId) return;

    if (confirm('Are you sure you want to delete this chat?')) {
        delete state.chats[state.currentChatId];

        // If no chats left, create a new one
        if (Object.keys(state.chats).length === 0) {
            createNewChat();
        } else {
            // Load the most recent chat
            const chatIds = Object.keys(state.chats);
            state.currentChatId = chatIds[chatIds.length - 1];
            loadChat(state.currentChatId);
        }

        updateChatHistoryList();
    }
}

async function sendMessage() {
    const message = elements.chatInput.value.trim();
    if (!message || !state.currentChatId) return;

    try {
        // Add loading indicator
        elements.sendMessageBtn.disabled = true;
        elements.sendMessageBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        
        // Add user message to UI
        addMessageToChat('user', message);
        elements.chatInput.value = '';

        // Send to backend and get response
        const result = await API.sendMessage(message, state.currentChatId);
        
        // Handle the response
        if (result) {
            // Debug: Log the structure of the response
            console.log('API Response structure:', result);
            
            // Extract the LLM response from the response object
            let assistantResponse;
            
            if (result.response && typeof result.response === 'object' && result.response.response) {
                // Handle nested response structure: { response: { response: "actual text" } }
                assistantResponse = result.response.response;
            } else if (result.response && typeof result.response === 'string') {
                // Handle direct string response: { response: "actual text" }
                assistantResponse = result.response;
            } else {
                // Handle unexpected response formats
                console.warn('Unexpected response format:', result);
                assistantResponse = "Sorry, I received an unexpected response format.";
            }
            
            // Add AI response to UI
            addMessageToChat('assistant', assistantResponse);
            
            // Update chat metadata if needed
            if (result.updated_title) {
                state.chats[state.currentChatId].title = result.updated_title;
                elements.currentChatTitle.textContent = result.updated_title;
                updateChatHistoryList();
            }
        }
    } catch (error) {
        console.error('Error sending message:', error); // Log the error details
        
        // Add error message to the chat
        addMessageToChat('assistant', `Error: ${error.message || 'Failed to get response'}`);
        
        // Alert the user
        alert(`Error sending message: ${error.message}`); 
    } finally {
        // Reset button state
        elements.sendMessageBtn.disabled = false;
        elements.sendMessageBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

function saveCurrentChat() {
    if (!state.currentChatId) return;

    // In a real app, this would save to a backend
    alert('Chat saved successfully!');
}

function updateChatHistoryList() {
    elements.chatHistory.innerHTML = '';

    // Sort chats by updatedAt (newest first)
    const sortedChats = Object.values(state.chats).sort((a, b) =>
        new Date(b.updatedAt) - new Date(a.updatedAt)
    );

    sortedChats.forEach(chat => {
        const li = document.createElement('li');
        li.dataset.chatId = chat.id;
        li.className = 'px-3 py-2 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center justify-between';
        li.innerHTML = `
            <div class="flex items-center truncate">
                <i class="fas fa-comment-alt text-gray-500 dark:text-gray-400 mr-2"></i>
                <span class="truncate dark:text-white">${chat.title}</span>
            </div>
        `;

        li.addEventListener('click', () => loadChat(chat.id));
        elements.chatHistory.appendChild(li);
    });

    // Highlight current chat
    if (state.currentChatId) {
        const currentChatItem = document.querySelector(`#chatHistory li[data-chat-id="${state.currentChatId}"]`);
        if (currentChatItem) {
            currentChatItem.classList.add('bg-blue-50', 'dark:bg-gray-700');
        }
    }
}

function addMessageToChat(role, content) {
    if (!state.currentChatId) return;

    const message = {
        role,
        content,
        timestamp: new Date().toISOString()
    };

    const chat = state.chats[state.currentChatId];
    chat.messages.push(message); // Ensure messages array is updated
    chat.updatedAt = new Date().toISOString();

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role} max-w-3/4 p-4 mb-4 w-fit`;

    if (role === 'user') {
        messageDiv.innerHTML = `
            <div class="flex items-end justify-end">
                <div class="flex flex-col space-y-2">
                    <div class="px-4 py-2">${content}</div>
                </div>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0 h-10 w-10 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center mr-3">
                    <i class="fas fa-robot text-gray-600 dark:text-gray-300"></i>
                </div>
                <div class="flex-1">
                    <div class="px-4 py-2">${content}</div>
                </div>
            </div>
        `;
    }

    elements.chatContainer.appendChild(messageDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

// Account Functions
function updateAccount() {
    const name = document.getElementById('accountName').value;
    const email = document.getElementById('accountEmail').value;
    const password = document.getElementById('accountPassword').value;

    if (!name || !email) {
        alert('Please fill in all required fields');
        return;
    }

    // In a real app, this would update the user's account on the backend
    state.currentUser.name = name;
    state.currentUser.email = email;
    elements.usernameDisplay.textContent = name;

    if (password) {
        alert('Password updated successfully');
    }

    elements.accountSettingsModal.classList.add('hidden');
}

// Admin Dashboard Functions
function renderAdminDashboard() {
    // Task Mapping Chart
    renderTaskMappingChart();

    // // User Satisfaction Chart
    // renderUserSatisfactionChart();

    // Word Cloud
    renderWordCloud();

    // ROI Heatmap
    renderROIHeatmap();

    // Topic Clustering
    renderTopicClustering();

    // Content Coverage Map
    renderContentCoverageMap();
}

async function renderTaskMappingChart() {
    // Define category labels for the chart with more descriptive names
    const labels = [
        'Summarization', 
        'Information Retrieval', 
        'Comparison', 
        'Extraction', 
        'Explanation', 
        'General'
    ];
    
    // Define corresponding data keys in the database
    const dataKeys = [
        'summarization', 
        'information_retrieval', 
        'comparison', 
        'extraction', 
        'explanation', 
        'general'
    ];
    
    // Define a vibrant color palette with good contrast
    const colorPalette = [
        '#3b82f6', // Blue
        '#10b981', // Green
        '#f59e0b', // Amber
        '#8b5cf6', // Purple
        '#64748b', // Slate
        '#ef4444'  // Red
    ];
    
    try {
        // Query Supabase conversation table
        const { data: categoryData, error } = await supabaseClient
            .from('conversation')
            .select('category')
            .neq('category', null);

        if (error) throw error;

        // Count category occurrences
        const counts = categoryData.reduce((acc, { category }) => {
            acc[category] = (acc[category] || 0) + 1;
            return acc;
        }, {});

        // Prepare Plotly data
        const values = dataKeys.map(key => counts[key] || 0);
        const total = values.reduce((sum, val) => sum + val, 0);
        
        // Create custom hover text with percentages and counts
        const hoverText = values.map((value, i) => {
            const percent = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
            return `${labels[i]}: ${value} (${percent}%)`;
        });
        
        const data = [{
            values: values,
            labels: labels,
            type: 'pie',
            textinfo: 'percent',
            hoverinfo: 'text',
            hovertext: hoverText,
            textfont: {
                size: 14,
                color: 'white'
            },
            marker: {
                colors: colorPalette,
                line: {
                    color: state.darkMode ? '#1e293b' : '#ffffff',
                    width: 2
                }
            },
            hole: 0.4, // Creates a donut chart for better visual appeal
            pull: values.map(v => v === Math.max(...values) ? 0.05 : 0), // Pull out the largest segment slightly
            direction: 'clockwise',
            rotation: 90,
            sort: false
        }];

        const layout = {
            height: 350,
            margin: {l: 10, r: 10, t: 0, b: 30},
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {
                color: state.darkMode ? '#f3f4f6' : '#1f2937',
                family: 'Arial, sans-serif'
            },
            legend: {
                font: {
                    size: 12,
                    color: state.darkMode ? '#f3f4f6' : '#1f2937'
                },
                orientation: 'h',
                xanchor: 'center',
                yanchor: 'bottom',
                y: -0.15,
                x: 0.5
            },
            annotations: [{
                font: {
                    size: 16,
                    color: state.darkMode ? '#f3f4f6' : '#1f2937'
                },
                showarrow: false,
                text: `Total: ${total}`,
                x: 0.5,
                y: 0.5
            }]
        };

        // Create responsive config
        const config = {
            responsive: true,
            displayModeBar: false,
            toImageButtonOptions: {
                format: 'png',
                filename: 'task_distribution_chart',
                height: 500,
                width: 700,
                scale: 2
            }
        };

        Plotly.newPlot('taskMappingChart', data, layout, config);
        
        // Add event listener for dark mode toggle to update chart colors
        elements.darkModeToggle.addEventListener('click', () => {
            setTimeout(() => {
                // Update layout colors based on current dark mode state
                const updatedLayout = {
                    title: {
                        font: {
                            color: state.darkMode ? '#f3f4f6' : '#1f2937'
                        }
                    },
                    font: {
                        color: state.darkMode ? '#f3f4f6' : '#1f2937'
                    },
                    legend: {
                        font: {
                            color: state.darkMode ? '#f3f4f6' : '#1f2937'
                        }
                    },
                    annotations: [{
                        font: {
                            color: state.darkMode ? '#f3f4f6' : '#1f2937'
                        },
                        showarrow: false,
                        text: `Total: ${total}`,
                        x: 0.5,
                        y: 0.5
                    }]
                };
                
                // Update the chart with new colors
                Plotly.relayout('taskMappingChart', updatedLayout);
                
                // Update marker line color
                const update = {
                    'marker.line.color': state.darkMode ? '#1e293b' : '#ffffff'
                };
                Plotly.restyle('taskMappingChart', update, 0);
            }, 100); // Small delay to ensure dark mode has toggled
        });
    } catch (error) {
        console.error('Error loading chart data:', error);
        
        // Display error message in the chart container
        document.getElementById('taskMappingChart').innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="text-center text-red-500">
                    <i class="fas fa-exclamation-triangle text-4xl mb-2"></i>
                    <p>Failed to load chart data</p>
                </div>
            </div>
        `;
    }
}

// ... existing code ...

// async function renderUserSatisfactionChart() {
//     try {
//         // Check if chart exists before destroying
//         if (window.userSatisfactionChart && typeof window.userSatisfactionChart.destroy === 'function') {
//             window.userSatisfactionChart.destroy();
//         }

//         // Get user satisfaction data from Supabase
//         const { data, error } = await supabaseClient
//             .from('user_feedback')
//             .select('rating, created_at')
//             .order('created_at', { ascending: true });

//         if (error) throw error;

//         // Process data for Chart.js
//         const dates = data.map(item => new Date(item.created_at).toLocaleDateString());
//         const ratings = data.map(item => item.rating);

//         const ctx = document.getElementById('userSatisfactionChart').getContext('2d');

//         window.userSatisfactionChart = new Chart(ctx, {
//             type: 'line',
//             data: {
//                 labels: dates,
//                 datasets: [{
//                     label: 'User Satisfaction',
//                     data: ratings,
//                     borderColor: '#3b82f6',
//                     backgroundColor: 'rgba(59, 130, 246, 0.1)',
//                     tension: 0.4,
//                     fill: true
//                 }]
//             },
//             options: {
//                 responsive: true,
//                 maintainAspectRatio: false,
//                 scales: {
//                     y: {
//                         beginAtZero: true,
//                         max: 5,
//                         ticks: {
//                             stepSize: 1,
//                             color: state.darkMode ? '#f3f4f6' : '#1f2937'
//                         },
//                         grid: {
//                             color: state.darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
//                         }
//                     },
//                     x: {
//                         ticks: {
//                             color: state.darkMode ? '#f3f4f6' : '#1f2937'
//                         },
//                         grid: {
//                             color: state.darkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
//                         }
//                     }
//                 },
//                 plugins: {
//                     legend: {
//                         labels: {
//                             color: state.darkMode ? '#f3f4f6' : '#1f2937'
//                         }
//                     }
//                 }
//             }
//         });
//     } catch (error) {
//         console.error('Error rendering user satisfaction chart:', error);
//     }
// }

async function renderWordCloud() {
    try {
        // Show loading state
        document.getElementById('wordCloud').innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
        `;
        
        // Fetch user messages from Supabase
        const { data, error } = await supabaseClient
            .from('conversation')
            .select('content')
            .eq('type', 'user');
            
        if (error) throw error;
        
        if (!data || data.length === 0) {
            document.getElementById('wordCloud').innerHTML = `
                <div class="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
                    <p>No user messages found</p>
                </div>
            `;
            return;
        }
        
        // Import stopwords
        const englishStopwords = window.stopwords?.en || [];
        const arabicStopwords = window.stopwords?.ar || [];
        
        // Combine stopwords
        const stopwords = [...englishStopwords, ...arabicStopwords];
        
        // Process text data
        const allText = data.map(item => item.content).join(' ');
        
        // Remove HTML tags
        const textWithoutHTML = allText.replace(/<[^>]*>/g, ' ');
        
        // Tokenize and clean text
        const words = textWithoutHTML
            .toLowerCase()
            .replace(/[^\w\s\u0600-\u06FF]/g, '') // Keep English and Arabic characters
            .split(/\s+/)
            .filter(word => word.length > 2) // Filter out short words
            .filter(word => !stopwords.includes(word)); // Filter out stopwords
        
        // Count word frequencies
        const wordFrequency = {};
        words.forEach(word => {
            wordFrequency[word] = (wordFrequency[word] || 0) + 1;
        });
        
        // Convert to array and sort by frequency
        const wordArray = Object.entries(wordFrequency)
            .map(([text, count]) => ({ text, size: Math.min(50, Math.max(10, count * 5)) }))
            .sort((a, b) => b.size - a.size)
            .slice(0, 50); // Limit to top 50 words
        
        // Clear the container
        document.getElementById('wordCloud').innerHTML = '';
        
        // Create word cloud
        WordCloud(document.getElementById('wordCloud'), {
            list: wordArray.map(word => [word.text, word.size]),
            backgroundColor: state.darkMode ? '#1e293b' : '#f8fafc',
            gridSize: 8,
            weightFactor: 2,
            fontFamily: 'sans-serif',
            color: function () {
                const colors = state.darkMode ?
                    ['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'] :
                    ['#1e40af', '#1e3a8a', '#1d4ed8', '#2563eb'];
                return colors[Math.floor(Math.random() * colors.length)];
            },
            rotateRatio: 0.3,
            rotationSteps: 3,
            drawOutOfBound: false,
            hover: function(item) {
                if (item) {
                    // Show tooltip
                    const tooltip = document.createElement('div');
                    tooltip.id = 'wordCloudTooltip';
                    tooltip.className = 'chart-tooltip';
                    tooltip.innerHTML = `${item[0]}: ${Math.floor(item[1]/5)} occurrences`;
                    tooltip.style.position = 'absolute';
                    tooltip.style.left = `${d3.event.pageX + 10}px`;
                    tooltip.style.top = `${d3.event.pageY + 10}px`;
                    document.body.appendChild(tooltip);
                } else {
                    // Remove tooltip
                    const tooltip = document.getElementById('wordCloudTooltip');
                    if (tooltip) tooltip.remove();
                }
            },
            click: function(item) {
                if (item) {
                    console.log(`Clicked on word: ${item[0]} with frequency: ${Math.floor(item[1]/5)}`);
                    // You could add functionality here to filter chats by this word
                }
            }
        });
    } catch (error) {
        console.error('Error rendering word cloud:', error);
        document.getElementById('wordCloud').innerHTML = `
            <div class="flex items-center justify-center h-full text-red-500">
                <p>Error loading word cloud: ${error.message}</p>
            </div>
        `;
    }
}

function renderROIHeatmap() {
    const departments = ['Sales', 'Marketing', 'Support', 'Engineering', 'Product'];
    const quarters = ['Q1', 'Q2', 'Q3', 'Q4'];

    const roiHeatmap = document.getElementById('roiHeatmap');
    roiHeatmap.innerHTML = '';

    departments.forEach(dept => {
        const row = document.createElement('tr');
        row.className = 'border-b border-gray-200 dark:border-gray-700';

        const deptCell = document.createElement('td');
        deptCell.className = 'py-3 text-sm font-medium dark:text-white';
        deptCell.textContent = dept;
        row.appendChild(deptCell);

        quarters.forEach(q => {
            const value = Math.floor(Math.random() * 100) + 1;
            const cell = document.createElement('td');
            cell.className = 'py-3 text-sm';

            const valueDiv = document.createElement('div');
            valueDiv.className = `heatmap-cell inline-flex items-center justify-center w-10 h-10 rounded-full text-xs font-semibold ${getHeatmapColor(value)}`;
            valueDiv.textContent = value;

            cell.appendChild(valueDiv);
            row.appendChild(cell);
        });

        roiHeatmap.appendChild(row);
    });
}

function getHeatmapColor(value) {
    if (value >= 80) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    if (value >= 60) return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
    if (value >= 40) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
}

function renderTopicClustering() {
    const ctx = document.getElementById('topicClusteringChart').getContext('2d');

    if (window.topicClusteringChart instanceof Chart) {
        window.topicClusteringChart.destroy();
    }

    window.topicClusteringChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Technical Support',
                    data: [{ x: 10, y: 20 }, { x: 15, y: 25 }, { x: 12, y: 18 }, { x: 8, y: 22 }, { x: 14, y: 19 }],
                    backgroundColor: '#3b82f6'
                },
                {
                    label: 'Account Management',
                    data: [{ x: 30, y: 40 }, { x: 35, y: 45 }, { x: 32, y: 38 }, { x: 28, y: 42 }, { x: 34, y: 39 }],
                    backgroundColor: '#10b981'
                },
                {
                    label: 'Billing',
                    data: [{ x: 50, y: 10 }, { x: 55, y: 15 }, { x: 52, y: 8 }, { x: 48, y: 12 }, { x: 54, y: 9 }],
                    backgroundColor: '#f59e0b'
                },
                {
                    label: 'Product Features',
                    data: [{ x: 20, y: 50 }, { x: 25, y: 55 }, { x: 22, y: 48 }, { x: 18, y: 52 }, { x: 24, y: 49 }],
                    backgroundColor: '#8b5cf6'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Topic Relevance',
                        color: state.darkMode ? '#f3f4f6' : '#1f2937'
                    },
                    grid: {
                        color: state.darkMode ? '#374151' : '#e5e7eb'
                    },
                    ticks: {
                        color: state.darkMode ? '#9ca3af' : '#6b7280'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'User Engagement',
                        color: state.darkMode ? '#f3f4f6' : '#1f2937'
                    },
                    grid: {
                        color: state.darkMode ? '#374151' : '#e5e7eb'
                    },
                    ticks: {
                        color: state.darkMode ? '#9ca3af' : '#6b7280'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: state.darkMode ? '#f3f4f6' : '#1f2937'
                    }
                }
            }
        }
    });
}

function renderContentCoverageMap() {
    const topics = [
        'Installation', 'Account Setup', 'Billing', 'API', 'Mobile App',
        'Desktop App', 'User Management', 'Permissions', 'Security', 'Integrations',
        'Troubleshooting', 'Error Codes', 'Performance', 'Scaling', 'Backups',
        'Data Export', 'Data Import', 'Customization', 'Themes', 'Notifications'
    ];

    const contentCoverageMap = document.getElementById('contentCoverageMap');
    contentCoverageMap.innerHTML = '';

    topics.forEach(topic => {
        const coverage = Math.floor(Math.random() * 100) + 1;
        const div = document.createElement('div');
        div.className = 'flex flex-col items-center';

        const coverageDiv = document.createElement('div');
        coverageDiv.className = `w-full h-24 rounded-lg flex items-center justify-center text-white font-bold mb-2 ${getCoverageColor(coverage)}`;
        coverageDiv.textContent = `${coverage}%`;

        const label = document.createElement('span');
        label.className = 'text-xs text-center text-gray-700 dark:text-gray-300';
        label.textContent = topic;

        div.appendChild(coverageDiv);
        div.appendChild(label);
        contentCoverageMap.appendChild(div);
    });
}

function getCoverageColor(value) {
    if (value >= 80) return 'bg-green-500 dark:bg-green-600';
    if (value >= 50) return 'bg-yellow-500 dark:bg-yellow-600';
    return 'bg-red-500 dark:bg-red-600';
}

// Add these lines if they don't exist
document.getElementById('loginForm').addEventListener('submit', (e) => {
    e.preventDefault();
    login();
});

document.getElementById('signupForm').addEventListener('submit', (e) => {
    e.preventDefault();
    signup();
});

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);