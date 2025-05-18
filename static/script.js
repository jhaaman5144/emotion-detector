document.addEventListener('DOMContentLoaded', function() {
    const emotionForm = document.getElementById('emotionForm');
    const textInput = document.getElementById('textInput');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultContainer = document.getElementById('resultContainer');
    const emotionLabel = document.getElementById('emotionLabel');
    const emotionDescription = document.getElementById('emotionDescription');
    const emotionChart = document.getElementById('emotionChart');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    // Handle form submission
    emotionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const text = textInput.value.trim();
        
        if (!text) {
            showError('Please enter some text to analyze');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.classList.remove('d-none');
        resultContainer.classList.add('d-none');
        errorMessage.classList.add('d-none');
        
        // Prepare form data
        const formData = new FormData();
        formData.append('text', text);
        
        // Send request to analyze endpoint
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Error analyzing text');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            loadingIndicator.classList.add('d-none');
            
            // Show results
            displayResults(data);
        })
        .catch(error => {
            // Hide loading indicator
            loadingIndicator.classList.add('d-none');
            
            // Show error message
            showError(error.message || 'An error occurred while analyzing the text.');
        });
    });
    
    // Function to display results
    function displayResults(data) {
        // Set emotion label with color
        emotionLabel.textContent = data.prediction;
        emotionLabel.style.backgroundColor = data.color;
        
        // Set emotion description
        emotionDescription.textContent = data.prediction.toLowerCase();
        
        // Set chart image
        emotionChart.src = `data:image/png;base64,${data.visualization}`;
        
        // Show result container
        resultContainer.classList.remove('d-none');
        
        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to show error
    function showError(message) {
        errorText.textContent = message;
        errorMessage.classList.remove('d-none');
    }
    
    // Add example texts for quick testing
    const exampleTexts = [
        "I'm so happy today!",
        "This makes me so sad",
        "I'm furious about what happened",
        "It's just an ordinary day, nothing special"
    ];
    
    // Function to set example text
    textInput.addEventListener('dblclick', function() {
        const randomExample = exampleTexts[Math.floor(Math.random() * exampleTexts.length)];
        textInput.value = randomExample;
    });
});
