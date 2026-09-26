// frontend/script.js
// Vanilla JS frontend logic for the Internship FAQ & AI Assistant.
// Uses native fetch() to call the FastAPI backend - no frameworks/libraries.

const API_URL = "http://127.0.0.1:8000/chat";

const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const errorBanner = document.getElementById("errorBanner");

function appendMessage(text, sender) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${sender === "user" ? "user-message" : "bot-message"}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  messageDiv.appendChild(bubble);
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  return bubble;
}

function appendLoadingBubble() {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message bot-message";
  messageDiv.id = "loadingMessage";

  const bubble = document.createElement("div");
  bubble.className = "bubble loading";
  bubble.innerHTML = "<span></span><span></span><span></span>";

  messageDiv.appendChild(bubble);
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeLoadingBubble() {
  const loadingMessage = document.getElementById("loadingMessage");
  if (loadingMessage) {
    loadingMessage.remove();
  }
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

async function sendMessage(message) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const message = userInput.value.trim();
  if (!message) return;

  appendMessage(message, "user");
  userInput.value = "";
  userInput.disabled = true;
  sendBtn.disabled = true;

  appendLoadingBubble();

  try {
    const data = await sendMessage(message);
    removeLoadingBubble();
    appendMessage(data.response, "bot");
  } catch (err) {
    removeLoadingBubble();
    showError(
      "Sorry, something went wrong reaching the assistant server. " +
        "Please make sure the FastAPI backend is running on http://127.0.0.1:8000. " +
        `(${err.message})`
    );
  } finally {
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
});