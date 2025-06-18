// This file contains JavaScript code for client-side functionality, handling user interactions and making requests to the server.

// Model classifier page
console.log('scripts.js loaded');
// Wait for the DOM to be fully loaded before executing scripts
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded event triggered');
// Get the loading screen element
    const form = document.getElementById('classification-form');
    const resultDiv = document.getElementById('result');
// Adding a submit-event listener to the form    
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const formData = new FormData(form); // Gather form data 
        fetch('/classify-image', { // Send the form data to the server
            method: 'POST',
            body: formData
        })
        .then(response => response.json()) // Recieive the response as JSON
        .then(data => {
        loadingScreen.style.display = 'block'; // Show the loading screen and show the result screen
        document.getElementById('resultScreen').style.display = 'block';
// If there is an error in the response, alert the user and reset the form
        if (data.error) { 
            alert(data.error);
            document.querySelector('.upload-container').style.display = 'block';
            document.getElementById('resultScreen').style.display = 'none';
            return;
        }
        // Display the uploaded image in the result section
        document.getElementById('resultImage').src = preview.src;
        document.getElementById('interpretationImage').src = preview.src;

        // Get predictions from the response
        let predictionList = data.prediction;
        // Prediction list can be an array or an object, ensure it's an array
        if (!Array.isArray(predictionList)) predictionList = [];
        // Fill the prediction list in the UI
        const listContainer = document.querySelector('.predictions ul');
        if (listContainer) {
            listContainer.innerHTML = '';
            if (predictionList.length > 0) {
                predictionList.forEach(item => {
                    let className, probability;
                    if (Array.isArray(item)) {
                        className = item[0];
                        probability = item[1];
                    } else if (typeof item === 'object') {
                        className = item.class_name || item[0];
                        probability = item.probability || item[1];
                    }
                    const listItem = document.createElement('li');
                    listItem.innerHTML = `<strong>${className}</strong>: ${(probability * 100).toFixed(1)}%`;
                    listContainer.appendChild(listItem);
                });
            } else {
                // If no predictions are available, display a message
                const listItem = document.createElement('li');
                listItem.textContent = 'No classifications available.';
                listContainer.appendChild(listItem);
            }
        }
         // --- Top 5 voorspellingen tonen onderaan het scherm ---
        const top5Container = document.getElementById('top5Predictions');
        if (top5Container && data.top5) {
            // Gebruik direct de top5 uit de backend
            top5Container.innerHTML = '<h3>Top 5 voorspellingen</h3><ol></ol>';
            const ol = top5Container.querySelector('ol');
            data.top5.forEach(item => {
                let className = item.class_name;
                let probability = item.probability;
                const li = document.createElement('li');
                li.innerHTML = `<strong>${className}</strong>: ${(probability * 100).toFixed(2)}%`;
                ol.appendChild(li);
            });
        }
        // --- Einde top 5 ---
    });
    // Radiologist Classifier page
    const addButton = document.getElementById('addClassification');
    if (addButton) {
        console.log('Add Classification button found');
        // Add a clcik event listener to the button
        addButton.addEventListener('click', function() {
            console.log('Add Classification button clicked');
            const container = document.getElementById('classificationContainer');
            if (container) {
                console.log('Classification container found');
                // Create a new classification group with a select dropdown
                const newGroup = document.createElement('div');
                newGroup.className = 'classification-group';
                newGroup.style.marginBottom = '20px';
                // Dropdown for classification options
                const select = document.createElement('select');
                select.style.width = '80%';
                select.style.padding = '10px';
                select.style.border = '1px solid #ccc';
                select.style.borderRadius = '5px';
                select.style.marginBottom = '10px';
                // Create options for the dropdown
                const options = [
                    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
                    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
                ];

                options.forEach(option => {
                    const opt = document.createElement('option');
                    opt.value = option;
                    opt.textContent = option;
                    select.appendChild(opt);
                });

                newGroup.appendChild(select);
                container.appendChild(newGroup);
            } else {
                console.error('Classification container not found');
            }
        });
    } else {
        console.error('Add Classification button not found');
    }
});