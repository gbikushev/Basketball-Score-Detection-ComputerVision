document.addEventListener('DOMContentLoaded', function() {
    const loginButton = document.getElementById('login-button');
    const registerButton = document.getElementById('register-button'); // Add this line
    const logoutButton = document.getElementById('logout-button');
    const usernameFrame = document.getElementById('username-frame');
    const usernameDisplay = document.getElementById('username-display');

    document.getElementById('logout-button').addEventListener('click', function() {
        // send a POST request to the server when the "logout" button is clicked
        fetch('http://127.0.0.1:8000/auth/jwt/logout', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            // request body (empty in your case)
            body: JSON.stringify({})
        })
        .then(response => {
            // handle the response from the server
            if (response.ok) {
                // redirect to the login page or another page after successful logout
                window.location.href = '/login'; // replace '/login' with the path to the login page
            } else {
                // handle the logout error
                console.log('Failed to log out');
                // here you can display an error message to the user
            }
        })
        .catch(error => {
            console.error('An error occurred:', error);
        });
    });

    fetch('/protected-route')
        .then(response => {
            if (response.ok) {
                return response.text();
            } else {
                throw new Error('Unauthorized');
            }
        })
        .then(username => {
            loginButton.style.display = 'none';
            registerButton.style.display = 'none';
            logoutButton.style.display = 'block';
            usernameFrame.style.display = 'flex'; // Display username frame
            usernameDisplay.textContent = username.replace(/"/g, ""); // Update username display
        })
        .catch(error => {
            loginButton.style.display = 'block';
            registerButton.style.display = 'block';
            logoutButton.style.display = 'none';
            usernameFrame.style.display = 'none'; // Hide username frame
            console.error('An error occurred:', error);
        });

    document.getElementById('login-button').addEventListener('click', function() {
        window.location.href = '/login';
    });
    
    // Add event listener for the register button
    document.getElementById('register-button').addEventListener('click', function() {
        window.location.href = '/register';
    });
    
    document.getElementById('home-button').addEventListener('click', function() {
        window.location.href = 'homepage/';
    });
    document.getElementById('about-button').addEventListener('click', function() {
        window.location.href = '/about';
    });
    document.getElementById('developers-button').addEventListener('click', function() {
        window.location.href = '/developers';
    });
    document.getElementById('blog-button').addEventListener('click', function() {
        window.location.href = '/blog';
    });
});
