(function (w, d) {
  var id = 'podscribe-capture',
    n = 'script';
  var e = d.createElement(n);
  e.id = id;
  e.async = true;
  e.src = 'https://d34r8q7sht0t9k.cloudfront.net/tag.js';
  var s = d.getElementsByTagName(n)[0];
  var visitorIdCookie = document.cookie
    .split(';')
    .map((row) => row.trim())
    .find((row) => row.startsWith('pplx.visitor-id='));
  var deviceId = visitorIdCookie ? visitorIdCookie.split('=')[1] : null;
  s?.parentNode?.insertBefore(e, s);
  e.addEventListener('load', function () {
    w.podscribe('init', {
      user_id: '0d2e093e-4b80-4b47-b890-7a84960409c1',
      advertiser: 'perplexityai',
    });
    w.podscribe('view', { device_id: deviceId });
    w.podscribe('init', {
      user_id: '249864f8-3071-7052-b053-7ddf7bec332e',
      advertiser: 'perplexityai',
    });
    w.podscribe('view', { device_id: deviceId });
  });
})(window, document);
