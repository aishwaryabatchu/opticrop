document.addEventListener('DOMContentLoaded', () => {
    // 1. Dark Mode / Theme Toggle
    const htmlElement = document.documentElement;
    const darkModeToggle = document.getElementById('darkModeToggle');
    const darkModeIcon = document.getElementById('darkModeIcon');

    // Read saved theme, default to light
    const currentTheme = localStorage.getItem('theme') || 'light';
    htmlElement.setAttribute('data-bs-theme', currentTheme);
    updateDarkModeIcon(currentTheme);

    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTheme = htmlElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
            htmlElement.setAttribute('data-bs-theme', targetTheme);
            localStorage.setItem('theme', targetTheme);
            updateDarkModeIcon(targetTheme);
        });
    }

    function updateDarkModeIcon(theme) {
        if (!darkModeIcon) return;
        if (theme === 'dark') {
            darkModeIcon.className = 'bi bi-sun-fill fs-5 text-warning';
        } else {
            darkModeIcon.className = 'bi bi-moon-stars-fill fs-5 text-success';
        }
    }

    // 2. Mobile Sidebar Toggle
    const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
    const sidebar = document.querySelector('.sidebar');
    if (sidebarToggleBtn && sidebar) {
        sidebarToggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('show');
        });
    }

    // 3. Bootstrap Form Validation & Loading Spinner
    const forms = document.querySelectorAll('.needs-validation');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingMessage = document.getElementById('loadingMessage');

    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            // Check password confirmation matching on registration form
            if (form.id === 'registerForm') {
                const password = document.getElementById('password');
                const confirmPassword = document.getElementById('confirm_password');
                const confirmFeedback = document.getElementById('confirmPasswordFeedback');
                
                if (password && confirmPassword && password.value !== confirmPassword.value) {
                    confirmPassword.setCustomValidity('Passwords do not match');
                    if (confirmFeedback) {
                        confirmFeedback.textContent = 'Passwords do not match.';
                    }
                } else if (confirmPassword) {
                    confirmPassword.setCustomValidity('');
                }
            }

            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            } else {
                // If form is valid and it's a prediction/retraining form, show loader
                if (form.classList.contains('show-loader')) {
                    const message = form.getAttribute('data-loader-msg') || 'Running agricultural calculations...';
                    if (loadingMessage) {
                        loadingMessage.textContent = message;
                    }
                    if (loadingOverlay) {
                        loadingOverlay.style.display = 'flex';
                    }
                }
            }

            form.classList.add('was-validated');
        }, false);
    });

    // 4. Initialize Tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (typeof bootstrap !== 'undefined') {
        [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
});
