// 全局通用工具
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// 弹窗控制
function openModal(id) {
    $(`#${id}`).style.display = 'flex';
}
function closeModal(id) {
    $(`#${id}`).style.display = 'none';
}
// 关闭弹窗遮罩点击
$$('.modal-mask').forEach(mask => {
    mask.addEventListener('click', e => {
        if (e.target === mask) mask.style.display = 'none';
    })
})
$$('.modal-close').forEach(btn => {
    btn.onclick = () => btn.closest('.modal-mask').style.display = 'none';
})

// Tab切换
$$('.tab-item').forEach(tab => {
    tab.onclick = () => {
        const type = tab.dataset.type;
        $$('.tab-item').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        $$('.tab-content').forEach(c => c.classList.remove('show'));
        $(`.tab-content[data-type="${type}"]`).classList.add('show');
    }
})

// 通用AJAX封装
async function request(url, method = "GET", data = null) {
    const opts = { method, headers: {} };
    if (method.toUpperCase() === 'POST') {
        opts.headers['Content-Type'] = 'application/x-www-form-urlencoded';
        if (data) {
            const params = new URLSearchParams();
            Object.keys(data).forEach(k => params.append(k, data[k]));
            opts.body = params;
        }
    }
    const res = await fetch(url, opts);
    return await res.json();
}

// 文件上传ajax
async function uploadFile(url, file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(url, { method: "POST", body: formData });
    return await res.json();
}

// 消息提示
function msg(text, time = 1500) {
    const div = document.createElement('div');
    div.style.cssText = `
        position:fixed;top:30px;left:50%;transform:translateX(-50%);
        background:rgba(0,0,0,0.7);color:#fff;padding:10px 20px;
        border-radius:8px;z-index:9999;
    `;
    div.innerText = text;
    document.body.appendChild(div);
    setTimeout(() => div.remove(), time);
}

// 分页渲染
function renderPagination(total, page, pageSize, callback) {
    const wrap = $('.pagination');
    wrap.innerHTML = '';
    const totalPage = Math.ceil(total / pageSize);
    if (totalPage <= 1) return;
    for (let i = 1; i <= totalPage; i++) {
        const btn = document.createElement('button');
        btn.innerText = i;
        if (i === page) btn.classList.add('active');
        btn.onclick = () => callback(i);
        wrap.appendChild(btn);
    }
}
