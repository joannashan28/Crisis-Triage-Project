// Disaster Tweet Triage: content script.
//
// Runs on the demo feed page. For every tweet on the page it:
//   1. sends the tweet text to the model server (POST /api/classify)
//   2. adds a colored label with the Stage 2 category and its confidence
//   3. moves tweets that Stage 1 called "not informative" into a separate
//      "Filtered out" tab, so they can still be reviewed instead of lost
//
// It also adds a bar of category tabs for filtering the feed, and an
// "Answer key" switch that shows the human annotators' label next to the
// model's, so failure cases can be pointed out during the demo.
//
// The model server is the same server that serves the feed page, so every
// request goes to this page's own origin.

const API_BASE = window.location.origin;
const BATCH_SIZE = 25;
const BATCH_DELAY_MS = 100;
const FILTERED_TAB = "filtered";
const ALL_TAB = "all";

const triage = {
  categories: [],
  categoryByLabel: {},
  filteredInfo: null,
  notActionableLabel: "",
  activeTab: ALL_TAB,
  showAnswerKey: false,
  queue: [],
  queueTimer: null,
  countsScheduled: false,
  bar: null,
  tabList: null,
  statusElement: null,
  tabButtons: {},
};


// ---------------------------------------------------------------------------
// Talking to the server
// ---------------------------------------------------------------------------

async function loadLabels() {
  let response = await fetch(API_BASE + "/api/labels");
  if (!response.ok) {
    throw new Error("The server answered " + response.status + " for /api/labels");
  }
  let data = await response.json();

  triage.categories = data.categories;
  triage.filteredInfo = data.filtered;
  triage.notActionableLabel = data.not_actionable_label;
  for (let category of data.categories) {
    triage.categoryByLabel[category.label] = category;
  }
}


async function classifyTweets(articles) {
  let tweets = [];
  for (let article of articles) {
    let textElement = article.querySelector(".tweet-text");
    tweets.push({ id: article.dataset.tweetId, text: textElement.textContent });
  }

  let response = await fetch(API_BASE + "/api/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tweets: tweets }),
  });
  if (!response.ok) {
    throw new Error("The server answered " + response.status + " for /api/classify");
  }
  let data = await response.json();
  return data.results;
}


// ---------------------------------------------------------------------------
// Queueing tweets as they appear
//
// Tweets arrive 25 at a time when the feed loads or "Show more" is clicked.
// We wait a moment and send them together, instead of one request per tweet.
// ---------------------------------------------------------------------------

function queueTweet(article) {
  if (article.dataset.triageState) {
    return;
  }
  article.dataset.triageState = "pending";
  addPendingBadge(article);
  triage.queue.push(article);

  if (triage.queueTimer === null) {
    triage.queueTimer = setTimeout(sendQueue, BATCH_DELAY_MS);
  }
}


async function sendQueue() {
  triage.queueTimer = null;

  while (triage.queue.length > 0) {
    let batch = triage.queue.splice(0, BATCH_SIZE);
    let results;
    try {
      results = await classifyTweets(batch);
    } catch (error) {
      for (let article of batch) {
        markFailed(article);
      }
      setStatus("Some tweets could not be labeled: " + error.message);
      continue;
    }

    for (let index = 0; index < batch.length; index++) {
      applyResult(batch[index], results[index]);
    }
    scheduleCountUpdate();
    applyActiveTab();
  }
}


function queueTweetsOnPage() {
  let articles = document.querySelectorAll("article[data-tweet-id]");
  for (let article of articles) {
    queueTweet(article);
  }
}


// ---------------------------------------------------------------------------
// Labels on each tweet
// ---------------------------------------------------------------------------

function findOrCreateBadge(article) {
  let badge = article.querySelector(".triage-badge");
  if (badge === null) {
    badge = document.createElement("div");
    badge.className = "triage-badge";
    let body = article.querySelector(".tweet-body");
    if (body === null) {
      body = article;
    }
    body.appendChild(badge);
  }
  badge.replaceChildren();
  return badge;
}


function addPendingBadge(article) {
  let badge = findOrCreateBadge(article);
  badge.classList.add("is-pending");
  badge.textContent = "Labeling...";
}


function markFailed(article) {
  article.dataset.triageState = "failed";
  let badge = findOrCreateBadge(article);
  badge.classList.remove("is-pending");
  badge.textContent = "Not labeled. Check that the server is running.";
}


function percent(probability) {
  return Math.round(probability * 100) + "%";
}


function applyResult(article, result) {
  article.dataset.triageState = "done";
  article.dataset.triageEndToEnd = result.end_to_end_label;

  let color;
  let badge = findOrCreateBadge(article);
  badge.classList.remove("is-pending");

  let dot = document.createElement("span");
  dot.className = "triage-dot";
  dot.setAttribute("aria-hidden", "true");

  let name = document.createElement("span");
  name.className = "triage-name";

  let confidence = document.createElement("span");
  confidence.className = "triage-confidence";

  if (result.informative) {
    let category = triage.categoryByLabel[result.category];
    article.dataset.triageGroup = result.category;
    color = category.color;
    name.textContent = category.display;
    confidence.textContent = percent(result.category_confidence) + " confident";
  } else {
    article.dataset.triageGroup = FILTERED_TAB;
    color = triage.filteredInfo.color;
    name.textContent = "Filtered out by Stage 1";
    let notInformative = 1 - result.prob_informative;
    confidence.textContent = percent(notInformative) + " sure it's not informative";
  }

  badge.title = "Stage 1: " + percent(result.prob_informative) + " chance this is informative";
  badge.appendChild(dot);
  badge.appendChild(name);
  badge.appendChild(confidence);

  article.style.setProperty("--triage-color", color);
  article.classList.add("triage-labeled");

  addAnswerKey(article, badge);
}


// The answer key row: what the annotators said, and whether the model agrees.
function addAnswerKey(article, badge) {
  let goldLabel = article.dataset.goldEndToEnd;
  if (!goldLabel) {
    return;
  }

  let answer = document.createElement("div");
  answer.className = "triage-answer";

  let goldText = document.createElement("span");
  goldText.textContent = "Annotators: " + describeGold(article);
  answer.appendChild(goldText);

  if (article.dataset.triageEndToEnd !== goldLabel) {
    article.classList.add("triage-mismatch");
    let warning = document.createElement("span");
    warning.className = "triage-disagree";
    warning.textContent = "Model and annotators disagree";
    answer.appendChild(warning);
  } else {
    article.classList.remove("triage-mismatch");
  }

  badge.appendChild(answer);
}


// Shows both of the annotators' labels, since end-to-end "no action needed"
// can mean "not informative" or "informative but no category".
function describeGold(article) {
  let informativeness = article.dataset.goldInformativeness;
  let humanitarian = article.dataset.goldHumanitarian;

  let parts = [];
  if (informativeness === "informative") {
    parts.push("Informative");
  } else {
    parts.push("Not informative");
  }

  let category = triage.categoryByLabel[humanitarian];
  if (category) {
    parts.push(category.display);
  } else if (humanitarian) {
    parts.push(humanitarian);
  }
  return parts.join(", ");
}


// ---------------------------------------------------------------------------
// The triage bar: category tabs, counts, and the answer key switch
// ---------------------------------------------------------------------------

function makeTabButton(tabId, text, color) {
  let button = document.createElement("button");
  button.type = "button";
  button.className = "triage-tab";
  button.dataset.tab = tabId;
  button.setAttribute("aria-pressed", "false");

  if (color) {
    let dot = document.createElement("span");
    dot.className = "triage-dot";
    dot.style.setProperty("--triage-color", color);
    dot.setAttribute("aria-hidden", "true");
    button.appendChild(dot);
  }

  let label = document.createElement("span");
  label.textContent = text;
  button.appendChild(label);

  let count = document.createElement("span");
  count.className = "triage-count";
  count.textContent = "0";
  button.appendChild(count);

  button.addEventListener("click", function () {
    selectTab(tabId);
  });

  triage.tabButtons[tabId] = button;
  return button;
}


function buildTriageBar() {
  let bar = document.createElement("section");
  bar.className = "triage-bar";
  bar.setAttribute("aria-label", "Triage filters");

  let heading = document.createElement("div");
  heading.className = "triage-heading";

  let title = document.createElement("h2");
  title.className = "triage-title";
  title.textContent = "Triage";
  heading.appendChild(title);

  let answerKeyLabel = document.createElement("label");
  answerKeyLabel.className = "triage-switch";
  let answerKeyBox = document.createElement("input");
  answerKeyBox.type = "checkbox";
  answerKeyBox.addEventListener("change", function () {
    triage.showAnswerKey = answerKeyBox.checked;
    document.body.classList.toggle("triage-show-answers", triage.showAnswerKey);
  });
  answerKeyLabel.appendChild(answerKeyBox);
  answerKeyLabel.appendChild(document.createTextNode(" Show answer key"));
  heading.appendChild(answerKeyLabel);

  bar.appendChild(heading);

  let tabList = document.createElement("div");
  tabList.className = "triage-tabs";
  tabList.appendChild(makeTabButton(ALL_TAB, "All informative", null));
  for (let category of triage.categories) {
    tabList.appendChild(makeTabButton(category.label, category.display, category.color));
  }
  tabList.appendChild(makeTabButton(FILTERED_TAB, "Filtered out", triage.filteredInfo.color));
  bar.appendChild(tabList);

  let status = document.createElement("p");
  status.className = "triage-status";
  status.setAttribute("aria-live", "polite");
  bar.appendChild(status);

  triage.bar = bar;
  triage.tabList = tabList;
  triage.statusElement = status;

  placeBar(bar);
  selectTab(ALL_TAB);
}


// Put the bar right under the timeline header, or at the top of the page if
// the page has no such header.
function placeBar(bar) {
  let header = document.querySelector(".timeline-header");
  if (header !== null) {
    header.insertAdjacentElement("afterend", bar);
  } else {
    document.body.prepend(bar);
  }
}


function setStatus(text) {
  if (triage.statusElement !== null) {
    triage.statusElement.textContent = text;
  }
}


function showConnectionError(error) {
  let bar = document.createElement("section");
  bar.className = "triage-bar triage-error";
  bar.textContent = "Can't reach the triage model at " + API_BASE
    + ". Start the server, then reload this page. (" + error.message + ")";
  placeBar(bar);
}


function selectTab(tabId) {
  triage.activeTab = tabId;
  for (let id in triage.tabButtons) {
    let button = triage.tabButtons[id];
    let isActive = id === tabId;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  }
  applyActiveTab();
}


// Decide whether one tweet belongs in the current tab.
function belongsInActiveTab(article) {
  let group = article.dataset.triageGroup;

  if (triage.activeTab === ALL_TAB) {
    // Everything except tweets Stage 1 filtered out (including tweets
    // still being labeled, so nothing disappears while it waits).
    return group !== FILTERED_TAB;
  }
  return group === triage.activeTab;
}


function applyActiveTab() {
  let articles = document.querySelectorAll("article[data-tweet-id]");
  for (let article of articles) {
    if (belongsInActiveTab(article)) {
      article.classList.remove("triage-hidden");
    } else {
      article.classList.add("triage-hidden");
    }
  }
}


function updateCounts() {
  triage.countsScheduled = false;

  let counts = {};
  for (let id in triage.tabButtons) {
    counts[id] = 0;
  }

  let articles = document.querySelectorAll("article[data-tweet-id]");
  let labeled = 0;
  let mismatches = 0;
  for (let article of articles) {
    let group = article.dataset.triageGroup;
    if (!group) {
      continue;
    }
    labeled = labeled + 1;
    if (group !== FILTERED_TAB) {
      counts[ALL_TAB] = counts[ALL_TAB] + 1;
    }
    if (group in counts) {
      counts[group] = counts[group] + 1;
    }
    if (article.classList.contains("triage-mismatch")) {
      mismatches = mismatches + 1;
    }
  }

  for (let id in triage.tabButtons) {
    let countElement = triage.tabButtons[id].querySelector(".triage-count");
    countElement.textContent = String(counts[id]);
  }

  if (labeled > 0) {
    setStatus(labeled + " tweets labeled on this page. The model and annotators disagree on "
      + mismatches + " of them.");
  } else {
    setStatus("");
  }
}


// Many tweets can change at once, so recount at most once per frame.
function scheduleCountUpdate() {
  if (triage.countsScheduled) {
    return;
  }
  triage.countsScheduled = true;
  requestAnimationFrame(updateCounts);
}


// ---------------------------------------------------------------------------
// Watching the page for new tweets
// ---------------------------------------------------------------------------

function watchForNewTweets() {
  let observer = new MutationObserver(function () {
    queueTweetsOnPage();
    applyActiveTab();
    scheduleCountUpdate();
  });
  observer.observe(document.body, { childList: true, subtree: true });
}


async function startTriage() {
  try {
    await loadLabels();
  } catch (error) {
    showConnectionError(error);
    return;
  }

  buildTriageBar();
  queueTweetsOnPage();
  watchForNewTweets();
}

startTriage();
