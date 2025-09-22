document.addEventListener('DOMContentLoaded', () => {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            // Toggle kelas 'active' pada hamburger dan menu
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }
});