Component({
  data: {
    selected: 0,
    list: [
      {
        pagePath: "/pages/index/index",
        text: "我的幼儿",
        icon: "👶",
        iconActive: "👶"
      },
      {
        pagePath: "/pages/notices/notices",
        text: "通知公告",
        icon: "🔔",
        iconActive: "🔔"
      },
      {
        pagePath: "/pages/growth/growth",
        text: "成长记录",
        icon: "📸",
        iconActive: "📸"
      },
      {
        pagePath: "/pages/monitor/monitor",
        text: "实时监控",
        icon: "📹",
        iconActive: "📹"
      }
    ]
  },
  methods: {
    switchTab(e) {
      const data = e.currentTarget.dataset;
      const url = this.data.list[data.index].pagePath;
      wx.switchTab({ url });
      this.setData({ selected: data.index });
    }
  }
})
