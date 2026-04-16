from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from playwright.async_api import Page, async_playwright

DEFAULT_HEADERS = {
    "accept-language": "en-US,en;q=0.9",
    "upgrade-insecure-requests": "1",
}
CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--disable-features=IsolateOrigins,site-per-process",
    "--disable-infobars",
    "--no-first-run",
    "--no-service-autorun",
    "--password-store=basic",
    "--use-mock-keychain",
    "--lang=en-US",
    "--disable-component-update",
    "--metrics-recording-only",
]
STEALTH_INIT_SCRIPT = """
// Override navigator.webdriver
Object.defineProperty(navigator, 'webdriver', {
  get: () => undefined,
});

// Override navigator.languages
Object.defineProperty(navigator, 'languages', {
  get: () => ['en-US', 'en'],
});

// Override navigator.plugins with complete plugin objects
Object.defineProperty(navigator, 'plugins', {
  get: () => [
    { name: 'Chrome PDF Plugin', suffix: 'pdf', description: 'Portable Document Format' },
    { name: 'Chrome PDF Viewer', suffix: '', description: 'PDF Viewer' },
    { name: 'Native Client', suffix: 'nexe', description: 'NaCl' },
  ],
});

// Override navigator.platform
Object.defineProperty(navigator, 'platform', {
  get: () => 'MacIntel',
});

// Override navigator.hardwareConcurrency
Object.defineProperty(navigator, 'hardwareConcurrency', {
  get: () => 8,
});

// Override navigator.deviceMemory
Object.defineProperty(navigator, 'deviceMemory', {
  get: () => 8,
});

// Override navigator.connection
Object.defineProperty(navigator, 'connection', {
  get: () => ({ effectiveType: '4g', rtt: 50, saveData: false }),
});

// Override navigator.userActivation
Object.defineProperty(navigator, 'userActivation', {
  get: () => ({ hasBeenActive: true, isActive: true }),
});

// Complete window.chrome object
window.chrome = window.chrome || {
  runtime: {
    onMessage: { addListener: () => {} },
    onInstalled: { addListener: () => {} },
  },
  app: {
    getCurrent: () => null,
  },
  csi: () => ({ onloadT: Date.now() }),
  loadTimes: () => ({
    navigationStart: Date.now(),
    firstPaintAfterLoad: Date.now(),
  }),
};

// Canvas fingerprinting bypass
const originalCanvasGetContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function(type, ...args) {
  const ctx = originalCanvasGetContext.call(this, type, ...args);
  if (type === '2d' && ctx) {
    const originalToDataURL = ctx.toDataURL;
    ctx.toDataURL = function(...args2) {
      return originalToDataURL.apply(this, args2);
    };
  }
  return ctx;
};

// WebGL fingerprinting bypass
const originalWebGLGetContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function(type, ...args) {
  const ctx = originalWebGLGetContext.call(this, type, ...args);
  if (type === 'webgl' && ctx) {
    const originalGetParameter = ctx.getParameter;
    ctx.getParameter = function(key) {
      if (key === 3744) return 'Intel Inc.';
      if (key === 3745) return 'Intel Iris Pro Graphics';
      return originalGetParameter.call(this, key);
    };
  }
  return ctx;
};

// AudioContext bypass
const AudioContext = window.AudioContext || window.webkitAudioContext;
if (AudioContext) {
  const originalAudio = AudioContext.prototype;
  AudioContext.prototype.createAnalyser = function(...args) {
    const analyser = originalAudio.createAnalyser.apply(this, args);
    analyser.getFloatValue = function() { return 0.5; };
    return analyser;
  };
}

// Sync window.outerWidth/Height with viewport
if (window.innerWidth === 1440) {
  Object.defineProperty(window, 'outerWidth', { get: () => 1440 });
  Object.defineProperty(window, 'outerHeight', { get: () => 1200 });
}

// Override navigator.mimeTypes
Object.defineProperty(navigator, 'mimeTypes', {
  get: () => [
    { type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format' },
    { type: 'application/x-google-chrome-pdf', suffixes: 'pdf', description: 'Chrome PDF' },
  ],
});

// Double-verify navigator.userAgent
Object.defineProperty(navigator, 'userAgent', {
  get: () => (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    + 'AppleWebKit/537.36 (KHTML, like Gecko) '
    + 'Chrome/124.0.0.0 Safari/537.36'
  ),
});

// Randomize timing to avoid detection
const originalNow = Date.now;
Date.now = function() {
  return originalNow.call(this) + Math.random() * 0.5;
};

// Override navigator.permissions.query for notifications
const originalQuery = window.navigator.permissions?.query;
if (originalQuery) {
  window.navigator.permissions.query = (parameters) => (
    parameters && parameters.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : originalQuery(parameters)
  );
}

// Override navigator.userAgentData for Chrome's User-Agent Client Hint API
const originalUserAgentData = Object.getOwnPropertyDescriptor(Navigator.prototype, 'userAgentData');
if (!originalUserAgentData) {
  Object.defineProperty(Navigator.prototype, 'userAgentData', {
    get: () => ({
      brands: [
        { brand: 'Chromium', version: '124' },
        { brand: 'Not A Brand', version: '99' },
      ],
      mobile: false,
      platform: 'macOS',
      getHighEntropyValues: function(hints) {
        return Promise.resolve({
          platform: 'macOS',
          mobile: false,
          architecture: 'x86_64',
          model: '',
          fullVersionList: [
            { brand: 'Chromium', version: '124.0.6367.60' },
            { brand: 'Not A Brand', version: '99.0.0.0' },
          ],
        });
      },
    }),
  });
}

// Override navigator.buildID (used for bot detection)
Object.defineProperty(navigator, 'buildID', {
  get: () => '20240415120000',
});

// Override navigator.cookieEnabled (ensure cookies appear enabled)
if (navigator.cookieEnabled !== true) {
  Object.defineProperty(navigator, 'cookieEnabled', {
    get: () => true,
  });
}

// Override window.external (used for extension detection)
window.external = window.external || {
  AddSearchProvider: function() {},
  IsSearchProviderInstalled: function() { return 0; },
};

// Override window.matchMedia to ensure consistent results
const originalMatchMedia = window.matchMedia;
window.matchMedia = function(query) {
  const result = originalMatchMedia.call(this, query);
  // Ensure consistent media query results
  return {
    ...result,
    matches: result.matches,
    media: result.media,
    onchange: null,
    addListener: function(fn) { /* no-op for bot detection */ },
    removeListener: function(fn) { /* no-op for bot detection */ },
    addEventListener: function(type, fn) { /* no-op for bot detection */ },
    removeEventListener: function(type, fn) { /* no-op for bot detection */ },
    dispatchEvent: function(event) { return true; },
  };
};

// Override window.screen to ensure consistent screen properties
const originalScreen = window.screen;
if (originalScreen) {
  Object.defineProperty(window.screen, 'availWidth', {
    get: () => 1440,
  });
  Object.defineProperty(window.screen, 'availHeight', {
    get: () => 1100,
  });
  Object.defineProperty(window.screen, 'colorDepth', {
    get: () => 24,
  });
  Object.defineProperty(window.screen, 'pixelDepth', {
    get: () => 24,
  });
}

// Override window.navigator.permissions for more permission types
const originalPermissionsQuery = window.navigator.permissions?.query;
if (originalPermissionsQuery) {
  window.navigator.permissions.query = function(parameters) {
    if (parameters && parameters.name === 'notifications') {
      return Promise.resolve({ state: Notification.permission });
    }
    if (parameters && parameters.name === 'geolocation') {
      return Promise.resolve({ state: 'granted' });
    }
    if (parameters && parameters.name === 'camera') {
      return Promise.resolve({ state: 'granted' });
    }
    if (parameters && parameters.name === 'microphone') {
      return Promise.resolve({ state: 'granted' });
    }
    return originalPermissionsQuery.call(this, parameters);
  };
}

// Override document.hidden for Page Visibility API
const originalHidden = Object.getOwnPropertyDescriptor(Document.prototype, 'hidden');
if (!originalHidden) {
  Object.defineProperty(Document.prototype, 'hidden', {
    get: () => false,
  });
}

// Override window.sessionStorage and localStorage to prevent storage-based detection
if (!window.sessionStorage) {
  window.sessionStorage = {
    getItem: function(key) { return null; },
    setItem: function(key, value) {},
    removeItem: function(key) {},
    clear: function() {},
    length: 0,
    key: function(index) { return null; },
  };
}
if (!window.localStorage) {
  window.localStorage = {
    getItem: function(key) { return null; },
    setItem: function(key, value) {},
    removeItem: function(key) {},
    clear: function() {},
    length: 0,
    key: function(index) { return null; },
  };
}

// Override performance.timing for consistent timing API
const originalPerformance = window.performance;
if (originalPerformance) {
  const originalTiming = originalPerformance.timing;
  if (originalTiming) {
    window.performance.timing = {
      ...originalTiming,
      navigationStart: originalTiming.navigationStart || Date.now() - 10000,
      domainLookupStart: originalTiming.domainLookupStart || Date.now() - 5000,
      domainLookupEnd: originalTiming.domainLookupEnd || Date.now() - 4000,
      connectStart: originalTiming.connectStart || Date.now() - 3000,
      connectEnd: originalTiming.connectEnd || Date.now() - 2000,
      requestStart: originalTiming.requestStart || Date.now() - 1000,
      responseStart: originalTiming.responseStart || Date.now() - 500,
      responseEnd: originalTiming.responseEnd || Date.now(),
    };
  }
}
"""

_BROWSER_HEADLESS = True


def set_browser_headless(headless: bool) -> None:
    global _BROWSER_HEADLESS
    _BROWSER_HEADLESS = headless


@asynccontextmanager
async def launch_browser() -> AsyncIterator[Page]:
    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            "/tmp/shopping_agent_profile",
            headless=_BROWSER_HEADLESS,
            args=CHROMIUM_ARGS,
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            extra_http_headers=DEFAULT_HEADERS,
            color_scheme="light",
            device_scale_factor=2,
            has_touch=False,
            is_mobile=False,
            java_script_enabled=True,
            timezone_id="America/New_York",
            viewport={"width": 1440, "height": 1100},
        )
        await context.add_init_script(STEALTH_INIT_SCRIPT)
        try:
            yield context.pages[0]
        finally:
            await context.close()
