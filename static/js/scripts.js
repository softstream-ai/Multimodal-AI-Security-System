document.addEventListener('DOMContentLoaded', function () {
    // Toggle Navigation
    const burger = document.querySelector('.burger');
    const navLinks = document.querySelector('.nav-links');

    // Toggle mobile navigation
    burger?.addEventListener('click', () => {
        navLinks?.classList.toggle('active');
        burger?.classList.toggle('toggle'); // Optional: Add animation to burger icon
    });

    // Camera Feed Simulation
    const alertMessage = document.getElementById('alert-message');
    const startDetectionButton = document.getElementById('start-detection');

    startDetectionButton?.addEventListener('click', () => {
        alertMessage?.classList.remove('hidden');
        setTimeout(() => {
            alertMessage?.classList.add('hidden');
        }, 5000);
    });

    // Scroll-based animations for fade-up effect
    const fadeElements = document.querySelectorAll('.fade-up');

    const fadeInSection = () => {
        fadeElements.forEach((element) => {
            const rect = element.getBoundingClientRect();
            if (rect.top <= window.innerHeight && rect.bottom >= 0) {
                element.classList.add('active'); // Add the active class to trigger animation
            }
        });
    };

    // Initial check on page load
    fadeInSection();
    window.addEventListener('scroll', fadeInSection);

    // Flash messages handling
    const flashMessages = document.querySelector('.flashes');
    if (flashMessages) {
        setTimeout(() => {
            flashMessages.remove();  // Hide flash messages after 3 seconds
        }, 3000);
    }

    // Modal functionality
    const modal = document.getElementById("loginModal");
    const btn = document.getElementById("tryItNow");
    const closeBtn = document.getElementsByClassName("close-btn")[0];

    btn?.addEventListener('click', function (event) {
        event.preventDefault();
        modal.style.display = "flex";
        modal.classList.add('fade-in');
    });

    closeBtn?.addEventListener('click', function () {
        modal.style.display = "none";
    });

    window.addEventListener('click', function (event) {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    });

    // Login/Signup Toggle
    const loginForm = document.getElementById("loginForm");
    const signupForm = document.getElementById("signupForm");
    const modalTitle = document.getElementById("modal-title");
    const toggleText = document.getElementById("toggle-text");
    const successMessage = document.createElement('p');
    successMessage.style.color = 'green';
    successMessage.style.fontWeight = 'bold';
    successMessage.classList.add('success-message');

    // Function to toggle between login and signup forms
    function toggleForms() {
        if (signupForm.style.display === "none") {
            signupForm.style.display = "block";
            loginForm.style.display = "none";
            modalTitle.innerHTML = "Sign Up";
            toggleText.innerHTML = "Already have an account? <a href='#' id='toggle-form'>Login</a>";
        } else {
            signupForm.style.display = "none";
            loginForm.style.display = "block";
            modalTitle.innerHTML = "Login";
            toggleText.innerHTML = "Don't have an account? <a href='#' id='toggle-form'>Sign up</a>";
        }
    }

    // Use event delegation for the dynamically replaced "toggle-form" links
    document.addEventListener('click', function (event) {
        if (event.target && event.target.id === 'toggle-form') {
            event.preventDefault();
            toggleForms();
            successMessage.remove();  // Clear the error or success message when toggling forms
        }
    });

    // Login form submission handling
    loginForm?.addEventListener('submit', function (event) {
        event.preventDefault();

        const formData = new FormData(loginForm);

        // Make API call for login
        fetch('/login', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const existingMessage = loginForm.querySelector('.success-message, .danger-message');
            if (existingMessage) {
                existingMessage.remove();
            }

            if (data.success) {
                // Store username and login status
                localStorage.setItem('username', data.username);
                localStorage.setItem('isLoggedIn', 'true');

                successMessage.textContent = "Login successful! Redirecting to home...";
                successMessage.classList.remove('danger-message');
                successMessage.classList.add('success-message');
                loginForm.appendChild(successMessage);

                setTimeout(() => {
                    modal.style.display = "none";
                    window.location.href = "/";  // Redirect to home page
                }, 2000);
            } else {
                successMessage.classList.remove('success-message');
                successMessage.classList.add('danger-message');
                successMessage.textContent = data.message || "Invalid email or password!";
                loginForm.appendChild(successMessage);
            }
        })
        .catch(error => {
            const existingMessage = loginForm.querySelector('.success-message, .danger-message');
            if (existingMessage) {
                existingMessage.remove();
            }

            successMessage.textContent = "An error occurred. Please try again.";
            successMessage.classList.remove('success-message');
            successMessage.classList.add('danger-message');
            loginForm.appendChild(successMessage);
        });
    });

    // Signup form submission handling
    signupForm?.addEventListener('submit', function (event) {
        event.preventDefault();

        const password = signupForm.querySelector('input[name="password"]').value;
        const confirmPassword = signupForm.querySelector('input[name="confirm_password"]').value;

        if (!successMessage.parentNode) {
            signupForm.insertBefore(successMessage, signupForm.firstChild);
        }

        if (password !== confirmPassword) {
            successMessage.classList.remove('success-message');
            successMessage.classList.add('danger-message');
            successMessage.textContent = "Passwords do not match!";
            return;
        }

        const formData = new FormData(signupForm);

        fetch('/signup', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                successMessage.classList.remove('danger-message');
                successMessage.classList.add('success-message');
                successMessage.textContent = "Account created successfully! Please log in.";

                setTimeout(() => {
                    toggleForms();
                    successMessage.remove();
                }, 3000);
            } else {
                successMessage.classList.remove('success-message');
                successMessage.classList.add('danger-message');
                successMessage.textContent = data.message || "Signup failed!";
            }
        })
        .catch(error => {
            successMessage.classList.remove('success-message');
            successMessage.classList.add('danger-message');
            successMessage.textContent = "An error occurred. Please try again.";
        });
    });

    // Profile handling variables
    const tryItNowBtn = document.getElementById("tryItNow");
    const profileMenu = document.getElementById("profileMenu");
    const dropdownMenu = document.getElementById("dropdownMenu");
    const profileIcon = document.getElementById("profileIcon");
    const logoutBtn = document.getElementById("logoutBtn");
    const alertSystem = document.getElementById("alert-system-id");
    const cameraFeed = document.getElementById("camera-feed");
    const detectioncameraFeed = document.getElementById("detection-camera-feed");
    const imgDataGathering = document.getElementById("image-data-gathering");

    // Function to check login status
    function checkLoginStatus() {
        const isLoggedIn = localStorage.getItem('isLoggedIn') === 'true';

        if (isLoggedIn) {
            imgDataGathering?.classList.remove("hidden");
            tryItNowBtn?.classList.add("hidden");
            cameraFeed?.classList.remove("hidden");
            detectioncameraFeed?.classList.remove("hidden");
            alertSystem?.classList.remove("hidden");
            profileMenu?.classList.remove("hidden");
            profileIcon?.classList.remove("hidden"); // Show profile icon when logged in
            const username = localStorage.getItem('username');
            dropdownMenu.querySelector('.dropdown-item.username').textContent = username; // Set username in dropdown
        } else {
            tryItNowBtn?.classList.remove("hidden");
            profileMenu?.classList.add("hidden");
            profileIcon?.classList.add("hidden"); // Hide profile icon when logged out
        }
    }

    // Toggle dropdown menu on profile icon click
    profileIcon?.addEventListener('click', function () {
        dropdownMenu.classList.toggle("hidden");
    });

    // Logout functionality
    logoutBtn?.addEventListener('click', function () {
        console.log('Logout button clicked');
        localStorage.setItem('isLoggedIn', 'false');
        localStorage.removeItem('username');

        // Update visibility
        imgDataGathering?.classList.add("hidden");
        alertSystem?.classList.add("hidden"); // Hide alert system on logout
        profileMenu.classList.add("hidden"); // Hide the profile menu
        profileIcon.classList.add("hidden"); // Hide the profile icon
        tryItNowBtn.classList.remove("hidden"); // Show the "Try It Now" button

        console.log('User logged out. Hiding profile menu and icon, showing "Try It Now" button.');

        // Redirect after a short delay to allow UI update
        setTimeout(() => {
            window.location.href = "/";
        }, 500);
    });

    // Initially check login status on page load
    checkLoginStatus();
});


document.getElementById('use-camera').addEventListener('click', () => {
    document.getElementById('person-input').classList.remove('hidden');
    document.getElementById('camera-feed-section').classList.remove('hidden');
    document.getElementById('upload-section').classList.add('hidden');
    startCameraFeed();  // Activate the camera feed
  });

  document.getElementById('upload-images').addEventListener('click', () => {
    document.getElementById('person-input').classList.remove('hidden');
    document.getElementById('camera-feed-section').classList.add('hidden');
    document.getElementById('upload-section').classList.remove('hidden');
  });



// Upload images
document.getElementById('submit-images-btn').addEventListener('click', () => {
    const personName = document.getElementById('person-name').value.trim();
    const files = document.getElementById('image-files').files;
    const responseMessageDiv = document.getElementById('upload-response-message');
    responseMessageDiv.classList.add('hidden'); // Hide message initially

    // Validate family member name and images separately
    if (!personName) {
        responseMessageDiv.textContent = "Please enter the family member's name before uploading images.";
        responseMessageDiv.className = "danger-message";
        responseMessageDiv.classList.remove('hidden');
        return;
    }
    
    if (files.length < 40) {
        responseMessageDiv.textContent = `A minimum of 40 images is required. You have uploaded ${files.length} image(s).`;
        responseMessageDiv.className = "danger-message";
        responseMessageDiv.classList.remove('hidden');
        return;
    }

    responseMessageDiv.classList.add('hidden');
    const formData = new FormData();
    for (const file of files) {
        formData.append('images', file);
    }
    formData.append('person_name', personName);

    fetch('/upload-images', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        // Display success or error message based on server response
        responseMessageDiv.textContent = data.message;
        responseMessageDiv.className = data.success ? 'success-message' : 'danger-message';
        responseMessageDiv.classList.remove('hidden');
    })
    .catch(error => {
        // Display error if API call fails
        responseMessageDiv.textContent = "An unexpected error occurred. Please try again.";
        responseMessageDiv.className = 'danger-message';
        responseMessageDiv.classList.remove('hidden');
        console.error("Upload error:", error);
    });
});

// Cancel button: Clear files, name input, message, and hide upload section
document.getElementById('cancel-upload-btn').addEventListener('click', () => {
    document.getElementById('image-files').value = ''; // Clear file input
    document.getElementById('person-name').value = ''; // Clear person name
    const responseMessageDiv = document.getElementById('upload-response-message');
    responseMessageDiv.textContent = ''; // Clear message content
    responseMessageDiv.classList.add('hidden'); // Hide message
    responseMessageDiv.classList.remove('success-message', 'danger-message'); // Clear background color
    document.getElementById('upload-section').classList.add('hidden'); // Hide upload section
});

// Add New Member button: Clear input fields and messages for new entries
document.getElementById('add-new-member-btn').addEventListener('click', () => {
    document.getElementById('person-name').value = ''; // Clear person name
    document.getElementById('image-files').value = ''; // Clear file input
    const responseMessageDiv = document.getElementById('upload-response-message');
    responseMessageDiv.textContent = ''; // Clear message content
    responseMessageDiv.classList.add('hidden'); // Hide message
    responseMessageDiv.classList.remove('success-message', 'danger-message'); // Clear background color
});




// ----------------------   ALERT SYSTEM  ----------------------------- ///

  document.getElementById('alert-config-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const senderEmail = document.getElementById('sender-email').value;
    const senderPassword = document.getElementById('sender-password').value;
    const receiverEmail = document.getElementById('receiver-email').value;
    const responseMessageDiv = document.getElementById('response-message');

    // Make API request to configure alert system
    fetch('/configure-alert-system', {
        method: 'POST',
        body: JSON.stringify({
            sender_email: senderEmail,
            sender_password: senderPassword,
            receiver_email: receiverEmail
        }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(res => res.json())
    .then(data => {
        // Displaying response message below the form
        responseMessageDiv.textContent = data.message;
        responseMessageDiv.style.color = data.success ? '#28a745' : '#dc3545';  // Success: Green, Error: Red
    })
    .catch(error => {
        // Handle unexpected errors
        responseMessageDiv.textContent = "An error occurred. Please try again.";
        responseMessageDiv.style.color = '#dc3545';  // Error: Red
    });
});


/////////////////////////////////  capture images ////////////////////////////////
const video = document.getElementById('video-feed');
const responseMessageDiv = document.getElementById('capture-response-message');
const personNameInput = document.getElementById('person-name');
const cameraFeedSection = document.getElementById('camera-feed-section');

// Event Listener for "Use Camera" Button
document.getElementById('use-camera').addEventListener('click', () => {
    document.getElementById('person-input').classList.remove('hidden');
    cameraFeedSection.classList.remove('hidden');
    startCameraFeed();
});

// Function to display messages based on API response
function displayMessage(message, className) {
    responseMessageDiv.textContent = message;
    responseMessageDiv.className = className;
    responseMessageDiv.classList.remove("hidden");
    setTimeout(() => {
        responseMessageDiv.classList.add("hidden"); // Auto-hide after a few seconds for better UX
    }, 5000);  // Adjust timeout as needed
}

// Camera feed variables
let imageCount = 0; // Number of images captured
const requiredImages = 10; // Set the required number of images
let stream; // To hold the camera stream

// Start camera feed
async function startCameraFeed() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        video.srcObject = stream;
    } catch (error) {
        console.error("Error accessing the camera:", error);
        displayMessage("Unable to access the camera. Please enable camera permissions.", "danger-message");
    }
}

// Capture image and send to API
document.getElementById('capture-image-btn').addEventListener('click', async () => {
    const personName = personNameInput.value.trim();
    if (!personName) {
        displayMessage("Please enter a family member's name.", "danger-message");
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
    const formData = new FormData();
    formData.append('image', imageBlob);
    formData.append('person_name', personName);

    fetch('/capture-image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const messageType = data.success ? 'success-message' : 'danger-message';
        displayMessage(`${data.message} Captured image ${imageCount + 1} out of ${requiredImages}.`, messageType);
        
        // Increment image count
        imageCount++;

        // Check if required number of images has been captured
        if (imageCount >= requiredImages) {
            displayMessage(`Successfully captured ${requiredImages} images. Turning off the camera.`, 'success-message');
            stopCamera();
            setTimeout(() => {
                displayMessage("Camera turned off.", 'success-message');
            }, 3000); // Adjust as needed
        }
    })
    .catch(error => {
        console.error("Capture error:", error);
        displayMessage("An error occurred. Please try again.", "danger-message");
    });
});

// Stop camera feed
document.getElementById('stop-camera-btn').addEventListener('click', () => {
    stopCamera();
    displayMessage(`Camera force stopped. Captured ${imageCount} images.`, 'danger-message');
});

// Function to stop camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop()); // Stop all tracks
        stream = null;
    }
    video.srcObject = null; // Clear video source
    imageCount = 0; // Reset image count for next session
}


//---------------------------- detection using camera feed ------------------- /////
  // Start detection
 // Start camera feed
// Start camera feed with improved error handling
// HTML Element References
// DOM Elements
const detectionVideo = document.getElementById('detection-video-feed');
const detectionCanvas = document.getElementById('detection-overlay');
const alertMessageResponse = document.getElementById('alert-message-response');
const startDetectionButton = document.getElementById('start-detection');
const stopDetectionButton = document.getElementById('stop-detection');
const ctx = detectionCanvas.getContext('2d');

// Variables for video stream and detection interval
let detectStream;
let detectionInterval;

// Function to display alert messages on the screen
function displayAlertMessage(message, className) {
    alertMessageResponse.textContent = message;
    alertMessageResponse.className = className;
    alertMessageResponse.classList.remove("hidden");
    setTimeout(() => {
        alertMessageResponse.classList.add("hidden");
    }, 5000);
}

// Function to start the detection video stream
async function startDetection() {
    try {
        detectStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        detectionVideo.srcObject = detectStream;
        detectionCanvas.width = detectionVideo.videoWidth;
        detectionCanvas.height = detectionVideo.videoHeight;
        startDetectionButton.classList.add('hidden');
        stopDetectionButton.classList.remove('hidden');

        // Start capturing and sending frames every 100ms
        detectionInterval = setInterval(captureAndSendFrame, 100);
    } catch (error) {
        console.error("Error accessing the camera:", error);
        displayAlertMessage("Unable to access the camera. Please enable camera permissions.", "danger-message");
    }
}

// Function to capture a frame from the video and send it to the server
async function captureAndSendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = detectionVideo.videoWidth;
    canvas.height = detectionVideo.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(detectionVideo, 0, 0, canvas.width, canvas.height);

    // Convert canvas to image blob
    const imageBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
    const formData = new FormData();
    formData.append('frame', imageBlob);

    // Send the image to the server for processing
    fetch('/detect-drones-and-faces', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayAlertMessage(data.message, data.success ? 'success-message' : 'danger-message');
        if (data.image) {
            const image = new Image();
            image.src = 'data:image/jpeg;base64,' + data.image;

            // Clear previous overlay
            ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
            
            // Draw bounding boxes and labels on the overlay canvas
            image.onload = () => {
                ctx.drawImage(image, 0, 0, detectionCanvas.width, detectionCanvas.height);
            };
        }
    })
    .catch(() => displayAlertMessage("An error occurred. Please try again.", "danger-message"));
}

// Function to stop the detection and video stream
function stopDetection() {
    if (detectStream) {
        detectStream.getTracks().forEach(track => track.stop()); // Stop all camera tracks
    }
    detectionVideo.srcObject = null;
    clearInterval(detectionInterval);

    // Update button visibility
    startDetectionButton.classList.remove('hidden');
    stopDetectionButton.classList.add('hidden');
    displayAlertMessage("Detection stopped.", "success-message");

    // Clear the overlay canvas when detection stops
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
}

// Event Listeners for starting and stopping detection
startDetectionButton.addEventListener('click', startDetection);
stopDetectionButton.addEventListener('click', stopDetection);


///////////////////////////// DASHBOARD //////////////////////////////////
const dashboardModal = document.getElementById("existingModal");
const dashboardLink = document.getElementById("dashboardLink");
const closeModalButton = document.getElementById("closeExistingModal");

const familyDetailsContainer = document.getElementById('familyDetails');
const settingsSection = document.getElementById('settings');
const settingsLink = document.getElementById('settingsLink');
const changeUsernameButton = document.getElementById('changeUsernameButton');
const changePasswordButton = document.getElementById('changePasswordButton');
const usernameChangeForm = document.getElementById('usernameChangeForm');
const passwordChangeForm = document.getElementById('passwordChangeForm');

// New elements for messages
const usernameChangeMessage = document.createElement('div');
const passwordChangeMessage = document.createElement('div');
usernameChangeMessage.id = 'usernameChangeMessage';
passwordChangeMessage.id = 'passwordChangeMessage';

usernameChangeForm.appendChild(usernameChangeMessage);
passwordChangeForm.appendChild(passwordChangeMessage);

// Open the dashboard modal and fetch family members
dashboardLink.addEventListener('click', function (event) {
    event.preventDefault();
    dashboardModal.style.display = "flex";
    familyDetailsContainer.style.display = "block";
    settingsSection.classList.add('hidden');
    fetchFamilyMembers();
});

// Close the dashboard modal
closeModalButton.addEventListener('click', function () {
    dashboardModal.style.display = "none";
});

// Toggle back to family details section
function showFamilyDetails() {
    familyDetailsContainer.style.display = "block";
    settingsSection.classList.add('hidden');
}

// Fetch family members from the API
function fetchFamilyMembers() {
    console.log("Fetching family members...");
    familyDetailsContainer.innerHTML = 'Loading family members...';
    fetch('/family_members_details', {
        method: 'GET',
        credentials: 'include'
    })
    .then(response => response.json())
    .then(data => {
        displayFamilyMembers(data.family_members);
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        familyDetailsContainer.innerHTML = 'Failed to load family members.';
    });
}

// Display family members
function displayFamilyMembers(members) {
    familyDetailsContainer.innerHTML = '';
    members.forEach(member => {
        const memberDiv = document.createElement('div');
        memberDiv.className = 'family-member';
        memberDiv.innerHTML = `<img src="${member.image_url}" alt="${member.name}" /><p>${member.name}</p>`;
        familyDetailsContainer.appendChild(memberDiv);
    });
}

// Toggle to settings
settingsLink.addEventListener('click', function (event) {
    event.preventDefault();
    familyDetailsContainer.style.display = "none";
    settingsSection.classList.remove('hidden');
    usernameChangeForm.classList.add('hidden');
    passwordChangeForm.classList.add('hidden');
});

// Show the username change form
changeUsernameButton.addEventListener('click', function () {
    usernameChangeForm.classList.toggle('hidden');
    passwordChangeForm.classList.add('hidden');
});

// Show the password change form
changePasswordButton.addEventListener('click', function () {
    passwordChangeForm.classList.toggle('hidden');
    usernameChangeForm.classList.add('hidden');
});

// Handle username change submission
document.getElementById('submitUsernameChange').addEventListener('click', function () {
    const newUsername = document.getElementById('newUsername').value;
    const loginPassword = document.getElementById('loginPassword').value;
    
    fetch('/update_username', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_username: newUsername, login_password: loginPassword })
    })
    .then(response => response.json())
    .then(data => {
        const usernameChangeMessage = document.getElementById('usernameChangeMessage');
        usernameChangeMessage.textContent = data.message;
        if (data.success) {
            usernameChangeMessage.classList.add('success-message');
            usernameChangeMessage.classList.remove('error-message');

            // Wait for 2 seconds before triggering the logout button click
            setTimeout(() => {
                document.getElementById('logoutBtn').click(); // Automatically click the logout button
            }, 2000);
        } else {
            usernameChangeMessage.classList.add('error-message');
            usernameChangeMessage.classList.remove('success-message');
        }
    })
    .catch(error => {
        console.error("Error updating username:", error);
        usernameChangeMessage.textContent = "An error occurred while updating username.";
        usernameChangeMessage.classList.add('error-message');
    });
});


// Handle password change submission
document.getElementById('submitPasswordChange').addEventListener('click', function () {
    const currentPassword = document.getElementById('currentPassword').value;
    const newPassword = document.getElementById('newPassword').value;
    
    fetch('/update_password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword })
    })
    .then(response => response.json())
    .then(data => {
        passwordChangeMessage.textContent = data.message;
        if (data.success) {
            passwordChangeMessage.classList.add('success-message');
            passwordChangeMessage.classList.remove('error-message');
        } else {
            passwordChangeMessage.classList.add('error-message');
            passwordChangeMessage.classList.remove('success-message');
        }
    })
    .catch(error => {
        console.error("Error updating password:", error);
        passwordChangeMessage.textContent = "An error occurred while updating password.";
        passwordChangeMessage.classList.add('error-message');
    });
});


const familyDetailsLink = document.getElementById('familyDetailsLink');

familyDetailsLink.addEventListener('click', function (event) {
    event.preventDefault();
    showFamilyDetails(); // Call the function to show family details
});

// Toggle back to family details section
function showFamilyDetails() {
    familyDetailsContainer.style.display = "block";  // Show family details
    settingsSection.classList.add('hidden');         // Hide settings section
}