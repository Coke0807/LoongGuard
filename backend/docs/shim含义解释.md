"Shim" 是一个在计算机科学和工程领域非常常见的术语。

## 核心含义

**Shim** 的本意是**垫片**或**楔形物**（物理上用于填充空隙、调整高度或对齐）。

在计算机领域，它被引申为：**一种轻量级的适配层或代理层**，用于连接两个原本不兼容的接口。

---

## 常见使用场景

### 1. **API / 兼容性垫片（最常见）**
当新旧 API 不兼容时，写一个 shim 来兼容旧代码。

```javascript
// 旧代码调用的是旧的 API
oldApi.doSomething();

// Shim：让旧代码在新环境下也能工作
const oldApi = {
  doSomething: () => {
    // 内部调用新的 API
    return newApi.performAction();
  }
};
```

### 2. **语言/运行时垫片**
比如 **Node.js** 中的 `core-js` 就是一个典型的 shim，它让旧版浏览器支持新版 JavaScript 特性。

```javascript
// 旧浏览器没有 Array.prototype.includes
// 通过 shim 添加这个方法
if (!Array.prototype.includes) {
  Array.prototype.includes = function(searchElement) {
    // ... 实现逻辑
  };
}
```

### 3. **系统/库集成**
在操作系统或框架中，shim 常用于：
- 拦截系统调用
- 转换数据格式
- 添加日志或监控
- 实现依赖注入

---

## 与类似概念的区别

| 术语 | 含义 | 与 Shim 的区别 |
| :--- | :--- | :--- |
| **Wrapper** | 包装器 | 更通用，可能增加额外功能 |
| **Adapter** | 适配器 | 通常用于接口转换，结构更复杂 |
| **Proxy** | 代理 | 控制访问，可能添加额外逻辑 |
| **Shim** | 垫片 | **最轻量**，通常只负责桥接，尽量透明 |

---

## 一句话总结

> **Shim 就是代码世界里的"垫片"**——薄、轻、透明，专门用来填平两个接口之间的"缝隙"，让原本合不上的东西能无缝衔接。

在你的 `opencode` 环境中，如果提到 "shim"，很可能是指某种用于兼容或桥接的轻量级代理模块。