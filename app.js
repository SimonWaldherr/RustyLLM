const languageButton = document.querySelector('[data-language]');
const copyButton = document.querySelector('[data-copy]');

function applyLanguage(language) {
  document.documentElement.lang = language;
  document.querySelectorAll('[data-de]').forEach((element) => {
    if (!element.dataset.en) element.dataset.en = element.textContent;
    element.textContent = language === 'de' ? element.dataset.de : element.dataset.en;
  });
  document.querySelectorAll('[data-i18n-html]').forEach((element) => {
    element.innerHTML = language === 'de' ? element.dataset.de : element.dataset.en;
  });
  document.title = language === 'de'
    ? 'RustyLLM — Lokale Inferenz, nachvollziehbar'
    : 'RustyLLM — Local inference, made inspectable';
  localStorage.setItem('rustyllm-language', language);
}

const savedLanguage = localStorage.getItem('rustyllm-language');
applyLanguage(savedLanguage || (navigator.language.startsWith('de') ? 'de' : 'en'));

languageButton.addEventListener('click', () => {
  applyLanguage(document.documentElement.lang === 'en' ? 'de' : 'en');
});

copyButton.addEventListener('click', async () => {
  const command = document.querySelector('.install code').textContent.trim();
  try {
    await navigator.clipboard.writeText(command);
    copyButton.textContent = document.documentElement.lang === 'de' ? 'Kopiert!' : 'Copied!';
    setTimeout(() => {
      copyButton.textContent = document.documentElement.lang === 'de' ? 'Kopieren' : 'Copy';
    }, 1500);
  } catch (_) {
    copyButton.textContent = '⌘C';
  }
});
