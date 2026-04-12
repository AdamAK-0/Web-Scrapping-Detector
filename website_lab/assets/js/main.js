
const PRODUCT_DATA = [
  {
    "slug": "aqua-sense-pro",
    "name": "AquaSense Pro Monitor",
    "category": "Monitoring",
    "price": "$249",
    "short": "AI-assisted water quality monitor with anomaly alerts and live dashboard views.",
    "badge": "Best Seller"
  },
  {
    "slug": "flow-guard-mini",
    "name": "FlowGuard Mini Controller",
    "category": "Control",
    "price": "$179",
    "short": "Compact irrigation controller for research plots and smart greenhouse pilots.",
    "badge": "Lab Favorite"
  },
  {
    "slug": "hydra-node-x",
    "name": "HydraNode X Sensor Kit",
    "category": "Sensors",
    "price": "$319",
    "short": "Modular edge sensor node for pH, conductivity, turbidity, and flow.",
    "badge": "Field Ready"
  },
  {
    "slug": "puremesh-filter",
    "name": "PureMesh Filtration Unit",
    "category": "Filtration",
    "price": "$399",
    "short": "Research-grade microfiltration unit designed for pilot studies and demonstrations.",
    "badge": "New"
  },
  {
    "slug": "river-vision-cam",
    "name": "RiverVision Inspection Cam",
    "category": "Imaging",
    "price": "$289",
    "short": "Portable vision unit for outlet, drain, and channel inspection workflows.",
    "badge": "Computer Vision"
  },
  {
    "slug": "eco-reservoir-kit",
    "name": "EcoReservoir Starter Kit",
    "category": "Education",
    "price": "$149",
    "short": "Hands-on classroom kit for modeling water flow, storage, and treatment concepts.",
    "badge": "Education"
  }
];

function toggleMenu() {
  const nav = document.getElementById('mobileNav');
  if (nav) nav.classList.toggle('open');
}

const CART_KEY = 'aquashield-cart';

function getCartItems() {
  try {
    return JSON.parse(localStorage.getItem(CART_KEY) || '[]');
  } catch (e) {
    return [];
  }
}

function saveCartItems(items) {
  localStorage.setItem(CART_KEY, JSON.stringify(items));
}

function addToCart(name, price) {
  const current = getCartItems();
  current.push({ name, price, addedAt: new Date().toISOString() });
  saveCartItems(current);
  updateCartCount();
  renderCartPage();
  alert(`${name} added to cart.`);
}

function removeFromCart(index) {
  const items = getCartItems();
  if (index < 0 || index >= items.length) return;
  items.splice(index, 1);
  saveCartItems(items);
  updateCartCount();
  renderCartPage();
}

function clearCart() {
  saveCartItems([]);
  updateCartCount();
  renderCartPage();
}

function updateCartCount() {
  const items = getCartItems();
  document.querySelectorAll('[data-cart-count]').forEach(el => el.textContent = items.length);
}

function renderCatalog() {
  const wrap = document.getElementById('catalogGrid');
  if (!wrap) return;
  const q = (document.getElementById('searchInput')?.value || '').trim().toLowerCase();
  const category = document.getElementById('categoryFilter')?.value || 'All';
  const filtered = PRODUCT_DATA.filter(p => {
    const matchesQ = !q || [p.name, p.category, p.short, p.badge].join(' ').toLowerCase().includes(q);
    const matchesC = category === 'All' || p.category === category;
    return matchesQ && matchesC;
  });
  if (!filtered.length) {
    wrap.innerHTML = '<div class="search-empty">No products matched your filters. Try another keyword or category.</div>';
    return;
  }
  wrap.innerHTML = filtered.map(p => `
    <article class="product-card">
      <img src="/assets/images/${p.slug}.svg" alt="${p.name}" />
      <div class="product-card-body">
        <span class="badge">${p.badge}</span>
        <h3>${p.name}</h3>
        <p>${p.short}</p>
        <div class="meta-row">
          <span>${p.category}</span>
          <strong>${p.price}</strong>
        </div>
        <div class="card-actions">
          <a class="btn btn-primary" href="/pages/products/${p.slug}.html">View details</a>
          <button class="btn btn-secondary" onclick="addToCart('${p.name.replace(/'/g, "\'")}', '${p.price}')">Add to cart</button>
        </div>
      </div>
    </article>`).join('');
}

function setupFAQ() {
  document.querySelectorAll('.faq-item button').forEach(btn => {
    btn.addEventListener('click', () => {
      btn.parentElement.classList.toggle('open');
    });
  });
}

function setupContactForm() {
  const form = document.getElementById('contactForm');
  if (!form) return;
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const name = form.querySelector('[name="name"]').value.trim();
    const email = form.querySelector('[name="email"]').value.trim();
    const msg = form.querySelector('[name="message"]').value.trim();
    const output = document.getElementById('formStatus');
    if (!name || !email || !msg) {
      output.textContent = 'Please fill in your name, email, and message before sending.';
      output.className = 'notice';
      return;
    }
    output.textContent = `Thank you, ${name}. Your message has been captured for the demo website.`;
    output.className = 'notice';
    form.reset();
  });
}

function setupSearchPage() {
  const page = document.getElementById('searchResults');
  if (!page) return;
  const params = new URLSearchParams(window.location.search);
  const q = (params.get('q') || '').trim().toLowerCase();
  document.getElementById('searchQueryLabel').textContent = q || 'all items';
  const matches = PRODUCT_DATA.filter(p => [p.name,p.category,p.short,p.badge].join(' ').toLowerCase().includes(q));
  if (!matches.length) {
    page.innerHTML = '<div class="search-empty">No search results were found for this keyword.</div>';
    return;
  }
  page.innerHTML = matches.map(p => `
    <article class="product-card">
      <img src="/assets/images/${p.slug}.svg" alt="${p.name}" />
      <div class="product-card-body">
        <span class="badge">${p.badge}</span>
        <h3>${p.name}</h3>
        <p>${p.short}</p>
        <div class="meta-row"><span>${p.category}</span><strong>${p.price}</strong></div>
        <a class="btn btn-primary" href="/pages/products/${p.slug}.html">Open product</a>
      </div>
    </article>`).join('');
}



function injectCartShortcut() {
  if (document.querySelector('.floating-cart')) return;
  const link = document.createElement('a');
  link.href = '/cart.html';
  link.className = 'floating-cart';
  link.setAttribute('aria-label', 'Open cart');
  link.innerHTML = '<span class="floating-cart-icon">🛒</span><span class="floating-cart-text">Cart</span><span class="floating-cart-count" data-cart-count>0</span>';
  document.body.appendChild(link);

  document.querySelectorAll('.desktop-nav, #mobileNav').forEach(nav => {
    if (!nav || nav.querySelector('[data-cart-link]')) return;
    const cartLink = document.createElement('a');
    cartLink.href = '/cart.html';
    cartLink.className = 'nav-link' + (window.location.pathname.endsWith('/cart.html') || window.location.pathname === '/cart.html' ? ' active' : '');
    cartLink.setAttribute('data-cart-link', 'true');
    cartLink.innerHTML = 'Cart <span class="nav-cart-pill" data-cart-count>0</span>';
    nav.appendChild(cartLink);
  });

  updateCartCount();
}

function renderCartPage() {
  const list = document.getElementById('cartItems');
  const empty = document.getElementById('cartEmpty');
  const summary = document.getElementById('cartSummary');
  const totalEl = document.getElementById('cartTotal');
  if (!list || !empty || !summary || !totalEl) return;

  const items = getCartItems();
  if (!items.length) {
    list.innerHTML = '';
    empty.hidden = false;
    summary.hidden = true;
    totalEl.textContent = '$0';
    return;
  }

  empty.hidden = true;
  summary.hidden = false;

  const total = items.reduce((sum, item) => {
    const value = Number(String(item.price).replace(/[^0-9.]/g, '')) || 0;
    return sum + value;
  }, 0);

  totalEl.textContent = `$${total}`;
  list.innerHTML = items.map((item, index) => `
    <article class="cart-item">
      <div>
        <h3>${item.name}</h3>
        <p class="muted">Added ${new Date(item.addedAt).toLocaleString()}</p>
      </div>
      <div class="cart-item-meta">
        <strong>${item.price}</strong>
        <button class="btn btn-secondary btn-small" type="button" onclick="removeFromCart(${index})">Remove</button>
      </div>
    </article>
  `).join('');
}

function setupHeaderSearch() {
  const form = document.getElementById('heroSearchForm');
  if (!form) return;
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = form.querySelector('input').value.trim();
    window.location.href = `/search.html?q=${encodeURIComponent(q)}`;
  });
}

document.addEventListener('DOMContentLoaded', () => {
  updateCartCount();
  renderCatalog();
  setupFAQ();
  setupContactForm();
  setupSearchPage();
  renderCartPage();
  setupHeaderSearch();
  document.getElementById('searchInput')?.addEventListener('input', renderCatalog);
  document.getElementById('categoryFilter')?.addEventListener('change', renderCatalog);
});
