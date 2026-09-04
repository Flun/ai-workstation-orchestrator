const DEFAULT_HOST = "192.168.1.243";
const DEFAULT_PORT = 8999;

const serverInput = document.getElementById("server");
const syncButton = document.getElementById("sync");
const statusBox = document.getElementById("status");
const syncXButton = document.getElementById("sync-x");
const xStatusBox = document.getElementById("x-status");

// Fill the server field from saved settings (default: 192.168.1.243)
chrome.storage.local.get({ server: DEFAULT_HOST }, (data) => {
  serverInput.value = (data.server || DEFAULT_HOST).trim();
});

function saveServer() {
  const value = (serverInput.value || "").trim() || DEFAULT_HOST;
  serverInput.value = value;
  chrome.storage.local.set({ server: value });
  return value;
}

// "192.168.1.243" -> "http://192.168.1.243:8999"
// "host.example:9000" -> "http://host.example:9000"
// "http://host:9000/" -> "http://host:9000"
function toBaseUrl(value) {
  let v = (value || "").trim();
  if (!v) v = DEFAULT_HOST;
  if (/^https?:\/\//i.test(v)) {
    return v.replace(/\/+$/, "");
  }
  if (/^[\w.-]+(:\d+)?$/i.test(v)) {
    if (!v.includes(":")) v += ":" + DEFAULT_PORT;
    return "http://" + v;
  }
  return "http://" + DEFAULT_HOST + ":" + DEFAULT_PORT;
}

// Chrome enforces CORS on extension pages. For a server address that is not
// covered by the static host_permissions in manifest.json, ask the user once
// for permission to that origin (the prompt appears when a sync is clicked).
async function ensureOriginAccess(baseUrl) {
  const origin = baseUrl + "/";
  const granted = await chrome.permissions.contains({ origins: [origin] });
  if (granted) return;
  const allowed = await chrome.permissions.request({ origins: [origin] });
  if (!allowed) {
    throw new Error(`서버 ${origin}에 대한 권한이 허용되지 않아 전송할 수 없습니다. (팝업의 [허용]을 누르세요)`);
  }
}

async function syncCookies() {
  syncButton.disabled = true;
  statusBox.textContent = "Chrome에서 Instagram 쿠키 읽는 중…";
  try {
    const apiRoot = toBaseUrl(saveServer());
    await ensureOriginAccess(apiRoot);
    const cookies = await chrome.cookies.getAll({ domain: "instagram.com" });
    if (!cookies.length) throw new Error("Instagram 쿠키가 없습니다. Chrome에서 Instagram에 로그인했는지 확인하세요.");
    const response = await fetch(`${apiRoot}/api/instagram-cookie-sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cookies })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "동기화 실패");
    statusBox.textContent = `${result.count}개 동기화 완료${result.has_session ? " · 로그인 세션 확인" : "\n주의: sessionid가 없어 로그인이 필요할 수 있습니다."}`;
  } catch (error) {
    statusBox.textContent = `실패: ${error.message}`;
  } finally {
    syncButton.disabled = false;
  }
}

async function syncXCookies() {
  syncXButton.disabled = true;
  xStatusBox.textContent = "Chrome에서 X/Twitter 쿠키 읽는 중…";
  try {
    const apiRoot = toBaseUrl(saveServer());
    await ensureOriginAccess(apiRoot);
    const [xCookies, twitterCookies] = await Promise.all([
      chrome.cookies.getAll({ domain: "x.com" }),
      chrome.cookies.getAll({ domain: "twitter.com" })
    ]);
    const seen = new Set();
    const cookies = [];
    for (const cookie of [...xCookies, ...twitterCookies]) {
      const key = `${cookie.domain}|${cookie.name}`;
      if (seen.has(key)) continue;
      seen.add(key);
      cookies.push(cookie);
    }
    if (!cookies.length) throw new Error("X/Twitter 쿠키가 없습니다. Chrome에서 x.com에 로그인했는지 확인하세요.");
    const response = await fetch(`${apiRoot}/api/x-cookie-sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cookies })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "동기화 실패");
    xStatusBox.textContent = `${result.count}개 동기화 완료${result.has_session ? " · 로그인 세션 확인" : "\n주의: auth_token이 없어 로그인이 필요할 수 있습니다."}`;
  } catch (error) {
    xStatusBox.textContent = `실패: ${error.message}`;
  } finally {
    syncXButton.disabled = false;
  }
}

syncButton.addEventListener("click", syncCookies);
syncXButton.addEventListener("click", syncXCookies);
