document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    fetch('/auth/jwt/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw response.json();
        }
        console.log('Login successful');
        window.location.href = '/homepage';
    })
    .catch(errorPromise => {
        errorPromise.then(error => {
            console.error('Error:', error);
            const errorMessage = error.detail ? `Error logging in: ${error.detail}` : `Unknown error: ${JSON.stringify(error)}`;
            alert(errorMessage);
        });
    });
});


document.querySelector('.toggle-password').addEventListener('click', function() {
    const passwordInput = document.getElementById('password');
    const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
    passwordInput.setAttribute('type', type);
    this.textContent = this.textContent === '👁️' ? '🚫' : '👁️';
});