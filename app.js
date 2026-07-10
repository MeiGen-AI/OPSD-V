import { bibtex, categoryMeta, metrics, project } from "./data.js?v=20260710-arxiv-glass-buttons";
import { cases } from "./cases.generated.js?v=20260706-readable-titles";

const PAGE_SIZE = 6;
const GALLERY_VIDEO_BASE_URL = "https://github.com/KumapowerLIU/KumapowerLIU.github.io/releases/download/videos-v1/";
const HIDDEN_CASE_SLUGS = new Set([
  "sf-movie-000004",
]);
const DEFAULT_ALL_FIRST_PAGE_SLUGS = [
  "ll-movie-000115",
  "sf-movie-000011",
  "ll-mei-000013",
  "sf-movie-000002",
  "ll-mei-000016",
  "sf-mei-000010",
];
const GALLERY_PRIORITY_SLUGS = [
  "ll-mei-000013",
  "ll-movie-000115",
  "ll-movie-000023",
  "ll-movie-000009",
  "ll-mei-000022",
  "ll-movie-000068",
  "ll-movie-000019",
  "ll-movie-000071",
  "ll-movie-000004",
  "sf-movie-000011",
  "sf-movie-000002",
  "sf-mei-000010",
  "sf-movie-000076",
  "sf-movie-000040",
  "sf-movie-000003",
  "sf-movie-000087",
  "sf-movie-000059",
  "sf-movie-000058",
  "sf-movie-000060",
];
const GALLERY_DEPRIORITY_SLUGS = [
  "ll-mei-000009",
  "ll-mei-000040",
  "ll-mei-000075",
  "ll-mei-000076",
  "ll-mei-000094",
  "ll-movie-000062",
  "sf-mei-000078",
  "sf-mei-000054",
  "sf-mei-000111",
  "sf-mei-000002",
  "sf-mei-000009",
  "sf-mei-000007",
  "sf-movie-000046",
  "sf-mei-000069",
  "sf-mei-000075",
];
const MOTIVATION_PROMPTS = {
  1: {
    title: "Aerial beach scene",
    prompt: "An aerial view of a beach scene on a clear day, with a bright blue sky and a few scattered white clouds. Crowds fill the beach: some people rest under umbrellas while others walk along the shoreline. The water is clear turquoise, the sand is pale beige, and tall buildings line the coast, including a prominent white building. Palm trees surround the beach as gentle waves wash onto the shore.",
  },
  2: {
    title: "Motorcyclist gesture",
    prompt: "A person wearing a black T-shirt with the text \"Bright Eyes\" and a white helmet with a reflective visor is riding a motorcycle. The background shows a suburban area with trees, houses, and power lines under a partly cloudy sky. The person raises their arms and then lowers them while riding.",
  },
  3: {
    title: "Bird in rocky stream",
    prompt: "A small bird with dark plumage, likely a crow or raven, is seen in a shallow, rocky stream surrounded by lush greenery. The bird is initially standing on a rock in the stream, then it moves through the water, splashing and creating ripples. The background consists of dense foliage and moss-covered rocks, with sunlight filtering through the trees and casting a warm glow on the scene. The bird continues to move through the water, occasionally flapping its wings and adjusting its position.",
  },
  4: {
    title: "Mountain village flyover",
    prompt: "Aerial views of a traditional village nestled in a mountainous region, featuring white buildings with black roofs. The village is surrounded by lush greenery and autumn-colored trees, with large trees displaying vibrant orange and yellow leaves. The village is densely packed with closely spaced buildings, some with small courtyards or gardens. The surrounding landscape includes rolling hills and fields, with a clear sky above. The video captures the village from various angles, showing the layout and architecture of the buildings, as well as the natural environment surrounding it.",
  },
  5: {
    title: "Forested road aerial view",
    prompt: "The video provides an aerial view of a forested area with a winding road cutting through it. The road is surrounded by dense greenery, with various shades of green indicating different types of trees and vegetation. A few houses are visible, with one having a dark roof and another with a lighter-colored roof. The houses are nestled among the trees, with driveways leading up to them. The road curves and bends, creating a path that weaves through the forest. The colors are primarily green from the trees, with the road appearing as a dark, linear path. The video captures the layout and structure of the area, including the road, houses, and surrounding forest.",
  },
};

const state = {
  backbone: "all",
  benchmark: "all",
  category: "all",
  page: 1,
};

const resultsGrid = document.querySelector("#results-grid");
const resultsSummary = document.querySelector("#results-summary");
const pagination = document.querySelector("#pagination");
const promptDialog = document.querySelector("#prompt-dialog");
const dialogKicker = document.querySelector("#dialog-kicker");
const dialogTitle = document.querySelector("#dialog-title");
const dialogPrompt = document.querySelector("#dialog-prompt");
const copyPromptButton = document.querySelector("#copy-prompt");
const compareDialog = document.querySelector("#compare-dialog");
const compareDialogKicker = document.querySelector("#compare-dialog-kicker");
const compareDialogTitle = document.querySelector("#compare-dialog-title");
const compareLabels = [
  document.querySelector("#compare-left-label"),
  document.querySelector("#compare-right-label"),
];
const compareVideos = [
  document.querySelector("#compare-base-video"),
  document.querySelector("#compare-ours-video"),
];
let activePrompt = "";
let compareWasFullscreen = false;

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderProjectMeta() {
  const authorBlock = document.querySelector("#authors");
  authorBlock.innerHTML = `
    <div class="author-list">
      ${project.authors.map((author) => `
        <span>
          ${renderAuthorName(author)}
          ${author.affiliations.map((id) => `<sup>${id}</sup>`).join("")}
          ${(author.notes || []).map((note) => `<sup>${escapeHtml(note)}</sup>`).join("")}
        </span>
      `).join("")}
    </div>
    <div class="affiliation-list">
      ${project.affiliations.map((affiliation) => `<span><sup>${affiliation.id}</sup>${escapeHtml(affiliation.name)}</span>`).join("")}
    </div>
    <div class="author-notes">
      ${project.authorNotes.map((note) => `<span><sup>${escapeHtml(note.symbol)}</sup>${escapeHtml(note.text)}</span>`).join("")}
    </div>
  `;
}

function renderAuthorName(author) {
  const name = escapeHtml(author.name);
  if (!author.url) return name;
  return `<a href="${escapeHtml(author.url)}" target="_blank" rel="noopener noreferrer">${name}</a>`;
}

const PROMPT_PREVIEW_OVERRIDES = new Map([
  ["ll-mei-000022", "A person in a long heavy coat walks through a foggy gothic street."],
  ["sf-movie-000002", "A classic cinematic movie trailer follows a space man crossing a vast salt desert."],
]);

function promptPreview(item) {
  if (PROMPT_PREVIEW_OVERRIDES.has(item.slug)) return PROMPT_PREVIEW_OVERRIDES.get(item.slug);
  const prompt = item.prompt;
  const clean = prompt.replace(/\s+/g, " ").trim();
  const firstSentenceMatch = clean.match(/^.+?[.!?](?:\s|$)/);
  const firstSentence = (firstSentenceMatch ? firstSentenceMatch[0] : clean).trim();
  if (firstSentence.length <= 140) return firstSentence;

  const naturalBreaks = [", with ", ", while ", ", featuring ", ", capturing ", ", wearing ", ", including ", ", and "];
  for (const token of naturalBreaks) {
    const index = firstSentence.indexOf(token);
    if (index >= 42 && index <= 128) {
      return punctuate(firstSentence.slice(0, index));
    }
  }
  const comma = firstSentence.lastIndexOf(",", 128);
  if (comma >= 58) return punctuate(firstSentence.slice(0, comma));

  const softLimit = 122;
  const cut = firstSentence.lastIndexOf(" ", softLimit);
  return punctuate(firstSentence.slice(0, cut > 72 ? cut : softLimit));
}

function punctuate(value) {
  const sentence = value.trim().replace(/[,:;]+$/u, "");
  return /[.!?]$/u.test(sentence) ? sentence : `${sentence}.`;
}

function resolveGalleryVideoUrl(path) {
  if (!path || /^https?:\/\//u.test(path)) return path;
  if (!path.startsWith("assets/videos/") || path.includes("/ablations/")) return path;
  return `${GALLERY_VIDEO_BASE_URL}${encodeURIComponent(path.split("/").pop())}`;
}

function makeCard(item) {
  const category = categoryMeta[item.category];
  const preview = promptPreview(item);
  const baseVideo = resolveGalleryVideoUrl(item.baseVideo);
  const oursVideo = resolveGalleryVideoUrl(item.oursVideo);
  const card = document.createElement("article");
  card.className = "result-card";
  card.dataset.backbone = item.backbone;
  card.dataset.category = item.category;
  card.dataset.case = item.slug;
  card.innerHTML = `
    <header class="card-head">
      <div>
        <div class="case-meta">${escapeHtml(item.backbone)} / ${escapeHtml(item.benchmark)}</div>
        <h3>${escapeHtml(item.title)}</h3>
      </div>
      <div class="card-actions">
        <span class="category-badge" data-category="${item.category}">${escapeHtml(category.short)}</span>
        <button class="compare-button" type="button" aria-label="Open side-by-side fullscreen comparison for ${escapeHtml(item.title)}">
          <span aria-hidden="true">&#x26F6;</span> Side-by-side fullscreen
        </button>
      </div>
    </header>
    <div class="video-pair" aria-label="Synchronized base and OPSD-V video comparison">
      <div class="video-cell">
        <span class="video-label">Base</span>
        <video muted loop playsinline controls preload="none"
          poster="${item.basePoster || `assets/posters/${item.slug}-base.jpg`}"
          data-src="${baseVideo}"
          aria-label="${escapeHtml(item.backbone)} base result for ${escapeHtml(item.title)}"></video>
        <span class="video-status">1 min</span>
      </div>
      <div class="video-cell is-ours">
        <span class="video-label">+ OPSD-V</span>
        <video muted loop playsinline controls preload="none"
          poster="${item.oursPoster || `assets/posters/${item.slug}-opsdv.jpg`}"
          data-src="${oursVideo}"
          aria-label="${escapeHtml(item.backbone)} with OPSD-V for ${escapeHtml(item.title)}"></video>
        <span class="video-status">1 min</span>
      </div>
    </div>
    <div class="prompt-preview">
      <p>${escapeHtml(preview)}</p>
      <button class="prompt-button" type="button" aria-label="Read full prompt for ${escapeHtml(item.title)}">Full prompt</button>
    </div>
  `;

  card.querySelector(".prompt-button").addEventListener("click", () => openPrompt(item));
  card.querySelector(".compare-button").addEventListener("click", () => openComparison(item));
  setupSynchronizedVideos(card);
  return card;
}

function setupSynchronizedVideos(card) {
  if (!card) return;
  const videos = [...card.querySelectorAll("video")];
  let syncing = false;

  const getPlaybackRate = (video) => {
    const configuredRate = Number(video.dataset.playbackRate);
    if (Number.isFinite(configuredRate) && configuredRate > 0) return configuredRate;
    return video.playbackRate || 1;
  };

  const syncPeerTime = (source, peer) => {
    const sourceTimelineTime = source.currentTime / getPlaybackRate(source);
    const peerTime = sourceTimelineTime * getPlaybackRate(peer);
    if (Number.isFinite(peerTime) && Math.abs(peer.currentTime - peerTime) > 0.2) {
      peer.currentTime = Math.min(peer.duration || peerTime, peerTime);
    }
  };

  videos.forEach((video) => {
    video.addEventListener("play", async () => {
      if (syncing) return;
      const peers = videos.filter((candidate) => candidate !== video);
      syncing = true;
      peers.forEach((peer) => syncPeerTime(video, peer));
      await Promise.allSettled(peers.map((peer) => peer.play()));
      syncing = false;
    });

    video.addEventListener("pause", () => {
      if (syncing) return;
      const peers = videos.filter((candidate) => candidate !== video);
      syncing = true;
      peers.forEach((peer) => peer.pause());
      syncing = false;
    });

    video.addEventListener("seeking", () => {
      if (syncing) return;
      const peers = videos.filter((candidate) => candidate !== video);
      syncing = true;
      if (Number.isFinite(video.currentTime)) peers.forEach((peer) => syncPeerTime(video, peer));
      syncing = false;
    });

    video.addEventListener("ratechange", () => {
      if (video.dataset.playbackRate) return;
      videos.forEach((peer) => {
        if (peer === video || peer.dataset.playbackRate) return;
        if (peer.playbackRate !== video.playbackRate) peer.playbackRate = video.playbackRate;
      });
    });
  });
}

function setupAblationVideos() {
  document.querySelectorAll(".ablation-video-pair").forEach(setupSynchronizedVideos);
  document.querySelectorAll(".ablation-compare-button").forEach((button) => {
    button.addEventListener("click", () => {
      const caseCard = button.closest(".ablation-video-case");
      const videos = [...caseCard.querySelectorAll("video")];
      const labels = [...caseCard.querySelectorAll(".video-label")].map((label) => label.textContent.trim());
      const sectionTitleEl = button.closest(".ablation-card").querySelector(".ablation-card-heading strong");
      const caseTitleEl = caseCard.querySelector("h3");
      const caseKickerEl = caseCard.querySelector(".ablation-video-case-head span");
      const sectionTitle = sectionTitleEl ? sectionTitleEl.textContent.trim() : "";
      const caseTitle = caseTitleEl ? caseTitleEl.textContent.trim() : "";
      const caseKicker = caseKickerEl ? caseKickerEl.textContent.trim() : "";
      openVideoComparison({
        kicker: caseKicker || "Ablation",
        title: formatComparisonTitle(sectionTitle, caseTitle),
        leftLabel: labels[0] || "Left",
        rightLabel: labels[1] || "Right",
        leftVideo: getVideoSource(videos[0]),
        rightVideo: getVideoSource(videos[1]),
      });
    });
  });
}

function setupMotivationCacheStudy() {
  const cases = [...document.querySelectorAll(".cache-diagnosis-case")];
  const previousButton = document.querySelector(".cache-page-arrow-prev");
  const nextButton = document.querySelector(".cache-page-arrow-next");
  const pageCount = document.querySelector(".cache-page-count");
  let activeIndex = 0;

  cases.forEach((caseCard) => setupSynchronizedVideos(caseCard.querySelector(".cache-video-pair")));

  const showCase = (index) => {
    activeIndex = (index + cases.length) % cases.length;
    cases.forEach((caseCard, caseIndex) => {
      const active = caseIndex === activeIndex;
      caseCard.classList.toggle("is-active", active);
      if (!active) {
        caseCard.querySelectorAll("video").forEach((video) => video.pause());
      } else {
        caseCard.querySelectorAll("video[data-src]").forEach(loadVideo);
      }
    });

    if (pageCount) pageCount.textContent = `${activeIndex + 1} / ${cases.length}`;
  };

  previousButton?.addEventListener("click", () => showCase(activeIndex - 1));
  nextButton?.addEventListener("click", () => showCase(activeIndex + 1));

  document.querySelectorAll(".cache-compare-button").forEach((button) => {
    button.addEventListener("click", () => {
      const caseCard = button.closest(".cache-diagnosis-case");
      const pair = caseCard.querySelector(".cache-video-pair");
      const videos = [...pair.querySelectorAll("video")];
      const labels = [...pair.querySelectorAll(".video-label")].map((label) => label.textContent.trim());
      const caseId = caseCard.querySelector(".cache-case-title span")?.textContent.trim() || "LongLive";
      const caseTitle = caseCard.querySelector("h4")?.textContent.trim() || "Cache diagnosis";
      openVideoComparison({
        kicker: "Training-free LongLive cache diagnosis",
        title: `${caseId} — ${caseTitle}`,
        leftLabel: labels[0] || "Original inference",
        rightLabel: labels[1] || "Data-cache inference",
        leftVideo: getVideoSource(videos[0]),
        rightVideo: getVideoSource(videos[1]),
      });
    });
  });

  document.querySelectorAll(".cache-prompt-button").forEach((button) => {
    button.addEventListener("click", () => openMotivationPrompt(button.dataset.cachePrompt));
  });

  if (cases.length) showCase(0);
}

function openMotivationPrompt(promptId) {
  const item = MOTIVATION_PROMPTS[promptId];
  if (!item) return;
  activePrompt = item.prompt;
  dialogKicker.textContent = "Training-free LongLive cache diagnosis";
  dialogTitle.textContent = item.title;
  dialogPrompt.textContent = item.prompt;
  promptDialog.showModal();
}

function formatComparisonTitle(sectionTitle, caseTitle) {
  const cleanSection = sectionTitle ? sectionTitle.replace(/[.。]+$/u, "") : "";
  return [cleanSection, caseTitle].filter(Boolean).join(" — ");
}

function renderCases() {
  const visibleCases = cases.filter((item) => {
    if (HIDDEN_CASE_SLUGS.has(item.slug)) return false;
    const backboneMatch = state.backbone === "all" || item.backbone === state.backbone;
    const benchmarkMatch = state.benchmark === "all" || item.benchmark === state.benchmark;
    const categoryMatch = state.category === "all" || item.category === state.category;
    return backboneMatch && benchmarkMatch && categoryMatch;
  }).sort(compareGalleryCases);

  resultsGrid.replaceChildren();
  if (!visibleCases.length) {
    resultsGrid.innerHTML = '<div class="empty-state">No case matches these filters.</div>';
    resultsSummary.textContent = "0 comparisons";
    pagination.replaceChildren();
    return;
  }

  const useBalancedAllLayout = state.backbone === "all"
    && state.benchmark === "all"
    && state.category === "all";
  const pages = useBalancedAllLayout
    ? buildDefaultAllPages(visibleCases)
    : chunkCases(visibleCases);
  const pageCount = pages.length;
  state.page = Math.min(state.page, pageCount);
  const pageStart = (state.page - 1) * PAGE_SIZE;
  const pageCases = pages[state.page - 1];

  pageCases.forEach((item) => resultsGrid.append(makeCard(item)));
  if (useBalancedAllLayout) {
    const longLiveCount = pageCases.filter((item) => item.backbone === "LongLive").length;
    const selfForcingCount = pageCases.length - longLiveCount;
    const benchmarks = new Set(pageCases.map((item) => item.benchmark));
    const pageLabel = benchmarks.size === 1 ? pageCases[0].benchmark : "Mixed benchmarks";
    resultsSummary.textContent = `${pageLabel} · ${longLiveCount} LongLive + ${selfForcingCount} Self-Forcing · Page ${state.page} of ${pageCount}`;
  } else {
    resultsSummary.textContent = `${pageStart + 1}-${Math.min(pageStart + PAGE_SIZE, visibleCases.length)} of ${visibleCases.length} curated comparisons`;
  }
  renderPagination(pageCount);
  observeVideos();
}

function compareGalleryCases(a, b) {
  const aPriority = GALLERY_PRIORITY_SLUGS.indexOf(a.slug);
  const bPriority = GALLERY_PRIORITY_SLUGS.indexOf(b.slug);
  if (aPriority !== -1 || bPriority !== -1) {
    if (aPriority === -1) return 1;
    if (bPriority === -1) return -1;
    return aPriority - bPriority;
  }

  const aDepriority = GALLERY_DEPRIORITY_SLUGS.indexOf(a.slug);
  const bDepriority = GALLERY_DEPRIORITY_SLUGS.indexOf(b.slug);
  if (aDepriority !== -1 || bDepriority !== -1) {
    if (aDepriority === -1) return -1;
    if (bDepriority === -1) return 1;
    return aDepriority - bDepriority;
  }

  return 0;
}

function buildDefaultAllPages(items) {
  const bySlug = new Map(items.map((item) => [item.slug, item]));
  const firstPage = DEFAULT_ALL_FIRST_PAGE_SLUGS
    .map((slug) => bySlug.get(slug))
    .filter(Boolean);
  const firstPageSlugs = new Set(firstPage.map((item) => item.slug));
  const remainingItems = items.filter((item) => !firstPageSlugs.has(item.slug));
  const remainingPages = buildBalancedAllPages(remainingItems);
  return firstPage.length ? [firstPage, ...remainingPages] : remainingPages;
}

function chunkCases(items) {
  const pages = [];
  for (let index = 0; index < items.length; index += PAGE_SIZE) {
    pages.push(items.slice(index, index + PAGE_SIZE));
  }
  return pages;
}

function buildBalancedAllPages(items) {
  const benchmarkOrder = ["MovieGenBench", "MeiBench"];
  const pagesByBenchmark = new Map(benchmarkOrder.map((benchmark) => [
    benchmark,
    buildBalancedBackbonePages(items.filter((item) => item.benchmark === benchmark)),
  ]));
  const pages = [];

  while (benchmarkOrder.some((benchmark) => pagesByBenchmark.get(benchmark).length)) {
    benchmarkOrder.forEach((benchmark) => {
      const benchmarkPages = pagesByBenchmark.get(benchmark);
      if (benchmarkPages.length) pages.push(benchmarkPages.shift());
    });
  }
  return pages;
}

function buildBalancedBackbonePages(items) {
  const longLive = items.filter((item) => item.backbone === "LongLive");
  const selfForcing = items.filter((item) => item.backbone === "Self-Forcing");
  const pages = [];

  while (longLive.length || selfForcing.length) {
    const leftColumn = longLive.splice(0, Math.min(3, longLive.length));
    const rightColumn = selfForcing.splice(0, Math.min(3, selfForcing.length));

    while (leftColumn.length + rightColumn.length < PAGE_SIZE && (longLive.length || selfForcing.length)) {
      if (selfForcing.length > longLive.length || !longLive.length) {
        rightColumn.push(selfForcing.shift());
      } else {
        leftColumn.push(longLive.shift());
      }
    }

    const page = [];
    const pairedRows = Math.min(leftColumn.length, rightColumn.length);
    for (let index = 0; index < pairedRows; index += 1) {
      page.push(leftColumn[index], rightColumn[index]);
    }
    page.push(...leftColumn.slice(pairedRows), ...rightColumn.slice(pairedRows));
    pages.push(page);
  }
  return pages;
}

function renderPagination(pageCount) {
  pagination.replaceChildren();
  if (pageCount <= 1) return;

  pagination.append(makePageButton("Previous", state.page - 1, state.page === 1, "pagination-direction"));

  const pageList = getPageList(pageCount, state.page);
  pageList.forEach((page) => {
    if (page === "ellipsis") {
      const ellipsis = document.createElement("span");
      ellipsis.className = "pagination-ellipsis";
      ellipsis.textContent = "...";
      pagination.append(ellipsis);
      return;
    }
    const button = makePageButton(String(page), page, false, "pagination-number");
    if (page === state.page) {
      button.classList.add("is-active");
      button.setAttribute("aria-current", "page");
    }
    pagination.append(button);
  });

  pagination.append(makePageButton("Next", state.page + 1, state.page === pageCount, "pagination-direction"));
}

function getPageList(pageCount, currentPage) {
  if (pageCount <= 7) return Array.from({ length: pageCount }, (_, index) => index + 1);
  const pages = [1];
  const rangeStart = Math.max(2, currentPage - 1);
  const rangeEnd = Math.min(pageCount - 1, currentPage + 1);
  if (rangeStart > 2) pages.push("ellipsis");
  for (let page = rangeStart; page <= rangeEnd; page += 1) pages.push(page);
  if (rangeEnd < pageCount - 1) pages.push("ellipsis");
  pages.push(pageCount);
  return pages;
}

function makePageButton(label, page, disabled, className) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `pagination-button ${className}`;
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", () => {
    state.page = page;
    renderCases();
    document.querySelector("#filter-bar").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  return button;
}

function observeVideos() {
  const videos = [...document.querySelectorAll("video[data-src]")];
  if (!("IntersectionObserver" in window)) {
    videos.forEach(loadVideo);
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      loadVideo(entry.target);
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "700px 0px" });

  videos.forEach((video) => observer.observe(video));
}

function loadVideo(video) {
  if (!video.dataset.src) return;
  video.src = video.dataset.src;
  video.preload = "metadata";
  if (video.dataset.playbackRate) {
    const playbackRate = Number(video.dataset.playbackRate);
    if (Number.isFinite(playbackRate) && playbackRate > 0) {
      const applyPlaybackRate = () => {
        video.playbackRate = playbackRate;
      };
      applyPlaybackRate();
      video.addEventListener("loadedmetadata", applyPlaybackRate, { once: true });
    }
  }
  delete video.dataset.src;
  video.load();
}

function setupFilters() {
  document.querySelectorAll(".filter-group").forEach((group) => {
    const filterName = group.dataset.filter;
    group.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        state[filterName] = button.dataset.value;
        state.page = 1;
        group.querySelectorAll("button").forEach((candidate) => {
          candidate.classList.toggle("is-active", candidate === button);
        });
        renderCases();
      });
    });
  });
}

function openPrompt(item) {
  activePrompt = item.prompt;
  dialogKicker.textContent = `${item.backbone} / ${item.benchmark}`;
  dialogTitle.textContent = item.title;
  dialogPrompt.textContent = item.prompt;
  promptDialog.showModal();
}

function openComparison(item) {
  openVideoComparison({
    kicker: `${item.backbone} / ${item.benchmark}`,
    title: item.title,
    leftLabel: "Base",
    rightLabel: "+ OPSD-V",
    leftVideo: resolveGalleryVideoUrl(item.baseVideo),
    rightVideo: resolveGalleryVideoUrl(item.oursVideo),
  });
}

function getVideoSource(video) {
  if (!video) return "";
  return video.currentSrc || video.src || video.dataset.src || "";
}

function openVideoComparison({ kicker, title, leftLabel, rightLabel, leftVideo, rightVideo }) {
  compareDialogKicker.textContent = kicker;
  compareDialogTitle.textContent = title;
  compareLabels[0].textContent = leftLabel;
  compareLabels[1].textContent = rightLabel;
  const sources = [leftVideo, rightVideo];
  compareVideos.forEach((video, index) => {
    video.pause();
    video.src = sources[index];
    video.currentTime = 0;
    video.load();
  });
  compareDialog.showModal();

  if (compareDialog.requestFullscreen) {
    compareDialog.requestFullscreen({ navigationUI: "hide" }).catch(() => {
      // The dialog itself remains viewport-sized when native fullscreen is unavailable.
    });
  }
}

function closeComparison() {
  const finishClose = () => {
    compareVideos.forEach((video) => {
      video.pause();
      video.removeAttribute("src");
      video.load();
    });
    if (compareDialog.open) compareDialog.close();
  };

  if (document.fullscreenElement === compareDialog && document.exitFullscreen) {
    compareWasFullscreen = false;
    document.exitFullscreen().finally(finishClose);
  } else {
    finishClose();
  }
}

function setupComparisonDialog() {
  setupSynchronizedVideos(compareDialog);
  document.querySelector("#compare-dialog-close").addEventListener("click", closeComparison);
  compareDialog.addEventListener("click", (event) => {
    if (event.target === compareDialog) closeComparison();
  });
  compareDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeComparison();
  });
  document.addEventListener("fullscreenchange", () => {
    if (document.fullscreenElement === compareDialog) {
      compareWasFullscreen = true;
    } else if (compareWasFullscreen && compareDialog.open) {
      compareWasFullscreen = false;
      closeComparison();
    }
  });
}

function setupPromptDialog() {
  document.querySelector("#dialog-close").addEventListener("click", () => promptDialog.close());
  promptDialog.addEventListener("click", (event) => {
    if (event.target === promptDialog) promptDialog.close();
  });
  copyPromptButton.addEventListener("click", () => copyText(activePrompt, copyPromptButton));
}

function renderMetrics() {
  const body = document.querySelector("#metrics-body");
  body.innerHTML = metrics.map((row) => `
    <tr class="${row.ours ? "is-ours" : ""}">
      <td class="method-cell">${escapeHtml(row.method)}</td>
      <td>${row.params}</td>
      <td>${row.nfe}</td>
      <td class="${row.best.includes("quality") ? "best-value" : ""}">${row.quality}</td>
      <td class="${row.best.includes("dynamics") ? "best-value" : ""}">${row.dynamics}</td>
      <td class="${row.best.includes("semantic") ? "best-value" : ""}">${row.semantic}</td>
    </tr>
  `).join("");
}

function setupHeroLinks() {
  [
    [document.querySelector("#paper-link"), project.paperUrl],
    [document.querySelector("#code-link"), project.codeUrl],
    [document.querySelector("#project-link"), project.projectUrl],
  ].forEach(([link, url]) => {
    if (!link || !url) return;
    link.href = url;
    link.classList.remove("is-placeholder");
    link.removeAttribute("aria-disabled");
    if (/^https?:\/\//u.test(url)) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
  });
}

async function copyText(text, button) {
  const original = button.textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
  } catch (error) {
    button.textContent = "Select and copy";
  }
  window.setTimeout(() => {
    button.textContent = original;
  }, 1500);
}

function setupCitation() {
  const bibtexElement = document.querySelector("#bibtex");
  const copyButton = document.querySelector("#copy-bibtex");
  bibtexElement.textContent = bibtex;
  copyButton.addEventListener("click", () => copyText(bibtex, copyButton));
}

function setupSectionReveal() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const sections = [...document.querySelectorAll("main .section")];
  sections.forEach((section) => section.classList.add("is-reveal-pending"));

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.08, rootMargin: "0px 0px -50px" });

  sections.forEach((section) => observer.observe(section));
}

renderProjectMeta();
renderCases();
renderMetrics();
setupAblationVideos();
setupMotivationCacheStudy();
setupFilters();
setupPromptDialog();
setupComparisonDialog();
setupCitation();
setupHeroLinks();
setupSectionReveal();
