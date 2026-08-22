const state = { messages: [] };
const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const sendButton = document.querySelector("#send");

function render() {
  messagesEl.replaceChildren(...state.messages.map((message) => {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;
    const role = document.createElement("div");
    role.className = "role";
    role.textContent = message.role;
    const content = document.createElement("div");
    content.textContent = message.content;
    article.append(role, content);
    return article;
  }));
}

async function checkHealth() {
  try {
    const response = await fetch("/ready");
    statusEl.textContent = response.ok ? "local model ready" : "local model offline";
    statusEl.classList.toggle("ready", response.ok);
  } catch {
    statusEl.textContent = "gateway offline";
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const prompt = document.querySelector("#prompt");
  const content = prompt.value.trim();
  if (!content) return;

  state.messages.push({ role: "user", content });
  const assistant = { role: "assistant", content: "" };
  state.messages.push(assistant);
  prompt.value = "";
  render();
  sendButton.disabled = true;

  const payload = {
    model: document.querySelector("#model").value.trim(),
    messages: state.messages.slice(0, -1),
    stream: true,
    temperature: Number(document.querySelector("#temperature").value),
    max_tokens: Number(document.querySelector("#maxTokens").value),
  };

  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.split("\n").find((item) => item.startsWith("data: "));
        if (!line || line === "data: [DONE]") continue;
        const data = JSON.parse(line.slice(6));
        assistant.content += data.choices?.[0]?.delta?.content || "";
        render();
      }
    }
  } catch (error) {
    assistant.content = `Request failed: ${error.message}`;
    render();
  } finally {
    sendButton.disabled = false;
  }
}

document.querySelector("#composer").addEventListener("submit", sendMessage);
document.querySelector("#clear").addEventListener("click", () => { state.messages = []; render(); });
checkHealth();
