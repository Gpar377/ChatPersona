const chatBox = document.getElementById("chatBox");

function addMessage(text, type) {
  const msg = document.createElement("div");
  msg.classList.add("message", type);
  msg.innerText = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Upload chat file
function uploadFile() {
  const file = document.getElementById("fileInput").files[0];

  if (!file) {
    alert("Select a file first");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  fetch("http://localhost:5000/upload", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(() => alert("Chat uploaded successfully"))
  .catch(() => alert("Upload failed"));
}

// Send message
function sendMessage() {
  const input = document.getElementById("message");
  const text = input.value.trim();

  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  fetch("http://localhost:5000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message: text })
  })
  .then(res => res.json())
  .then(data => addMessage(data.reply, "ai"))
  .catch(() => addMessage("Error connecting to server", "ai"));
}