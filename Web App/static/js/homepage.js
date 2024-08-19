// js code specifically for homepage

document.addEventListener('DOMContentLoaded', function() {
    const freeThrowsButton = document.getElementById('free-throws-button');
    const threePointThrowsButton = document.getElementById('three-point-throws-button');
    const analyzeButton = document.getElementById('analyze-button');

    // Function to handle button clicks
    function handleButtonClick(url) {
        window.location.href = url;
    }

    // Check if user is authenticated
    fetch('/protected-route')
        .then(response => {
            if (response.ok) {
                // If user is authenticated, allow button clicks
                freeThrowsButton.addEventListener('click', function() {
                    handleButtonClick('/free-throws');
                });
                threePointThrowsButton.addEventListener('click', function() {
                    handleButtonClick('/three-point-throws');
                });
                analyzeButton.addEventListener('click', function() {
                    handleButtonClick('/dashboard-redirect');
                });
            } else {
                throw new Error('Unauthorized');
            }
        })
        .catch(error => {
            // If user is not authenticated, disable buttons and display an alert
            freeThrowsButton.addEventListener('click', function() {
                alert('You must be logged in to access this section.');
            });
            threePointThrowsButton.addEventListener('click', function() {
                alert('You must be logged in to access this section.');
            });
            analyzeButton.addEventListener('click', function() {
                alert('You must be logged in to access this section.');
            });
            console.error('An error occurred:', error);
        });
});
