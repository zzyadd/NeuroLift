// wordcloud.js - Handles word cloud visualization

// This function will be called to render the word cloud
async function renderWordCloud(containerId, textData) {
    try {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container with ID ${containerId} not found`);
            return;
        }

        // Show loading indicator
        container.innerHTML = `
            <div class="loading">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                <p class="mt-2 text-gray-500 dark:text-gray-400">Generating word cloud...</p>
            </div>
        `;

        // Use both English and Arabic stopwords from stopwords-iso
        const englishStopwords = window.stopwords ? window.stopwords.en : [];
        const arabicStopwords = window.stopwords ? window.stopwords.ar : [];
        const stopwords = [...englishStopwords, ...arabicStopwords];
        
        if (!stopwords || stopwords.length === 0) {
            throw new Error("Stopwords not available");
        }

        // Process the text data
        const words = processText(textData, stopwords);
        
        if (words.length === 0) {
            container.innerHTML = `
                <div class="flex items-center justify-center h-full">
                    <p class="text-gray-500 dark:text-gray-400">Not enough data to generate word cloud</p>
                </div>
            `;
            return;
        }

        // Clear the container
        container.innerHTML = '';

        // Set up the word cloud
        const width = container.offsetWidth;
        const height = container.offsetHeight;

        // Create SVG
        const svg = d3.select(`#${containerId}`)
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", `translate(${width/2},${height/2})`);

        // Set up the layout
        const layout = d3.layout.cloud()
            .size([width, height])
            .words(words.map(d => ({ text: d.text, size: d.size })))
            .padding(5)
            .rotate(() => ~~(Math.random() * 2) * 90)
            .fontSize(d => d.size)
            .on("end", draw);

        // Start the layout
        layout.start();

        // Function to draw the word cloud
        function draw(words) {
            // Color scale
            const color = d3.scaleOrdinal(d3.schemeCategory10);

            svg.selectAll("text")
                .data(words)
                .enter().append("text")
                .style("font-size", d => `${d.size}px`)
                .style("fill", (d, i) => color(i))
                .attr("text-anchor", "middle")
                .attr("transform", d => `translate(${d.x},${d.y}) rotate(${d.rotate})`)
                .text(d => d.text)
                .on("mouseover", function() {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .style("font-size", d => `${d.size * 1.2}px`);
                })
                .on("mouseout", function() {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .style("font-size", d => `${d.size}px`);
                });
        }
    } catch (error) {
        console.error("Error loading word cloud:", error);
        document.getElementById(containerId).innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle text-4xl mb-2"></i>
                <p>Error loading word cloud: ${error.message}</p>
            </div>
        `;
    }
}

// Process text to get word frequencies
function processText(text, stopwords) {
    if (!text) return [];
    
    // Convert to lowercase and split into words
    const words = text.toLowerCase()
        .replace(/[^\w\s\u0600-\u06FF]/g, '') // Allow English and Arabic characters
        .split(/\s+/)
        .filter(word => {
            // Use combined stopwords list
            const isStopword = stopwords.includes(word);
            return word.length > 2 && !isStopword;
        });
    
    // Calculate word frequencies
    const wordFreq = words.reduce((acc, word) => {
        acc[word] = (acc[word] || 0) + 1;
        return acc;
    }, {});

    // Convert to array and sort by frequency
    const wordArray = Object.entries(wordFreq)
        .map(([text, value]) => ({ text, size: 10 + value * 3 }))
        .sort((a, b) => b.size - a.size)
        .slice(0, 100);
    
    return wordArray;
}

// Make functions available globally
window.wordCloudUtils = {
    renderWordCloud,
    processText
};