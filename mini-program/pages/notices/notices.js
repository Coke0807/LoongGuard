const request = require("../../utils/request")
Page({
  data: {
    noticeList: [],
    filter: "all",
    displayList: [],
    unreadCount: 0
  },

  async onLoad() {
    await this.loadNotices()
  },

  onShow() {
    if (typeof this.getTabBar === 'function' && this.getTabBar()) {
      this.getTabBar().setData({ selected: 1 })
    }
  },

  async loadNotices() {
    try{
      const res = await request("/get-notice-list", {})
      if (res.code === 0) {
        this.setData({ noticeList: res.data })
        this.applyFilter()
      }
    }catch(e){
      wx.showToast({title:"加载失败", icon:"none"})
    }
  },

  switchFilter(e) {
    const filter = e.currentTarget.dataset.filter
    this.setData({ filter })
    this.applyFilter()
  },

  applyFilter() {
    const { noticeList, filter } = this.data
    let displayList
    if (filter === "unread") {
      displayList = noticeList.filter(item => !item.read)
    } else {
      displayList = [...noticeList]
    }
    const unreadCount = noticeList.filter(item => !item.read).length
    this.setData({ displayList, unreadCount })
  },

  async tapNotice(e) {
    const id = e.currentTarget.dataset.id
    const item = this.data.noticeList.find(n => n.id === id)
    if (!item) return

    if (!item.read) {
      try{
        const res = await request("/read-notice", { notice_id: id })
        if (res.code === 0) {
          item.read = true
          this.setData({ noticeList: this.data.noticeList })
          this.applyFilter()
        }
      }catch(err){
        wx.showToast({title:"标记已读失败", icon:"none"})
      }
    }

    wx.showModal({
      title: item.title,
      content: item.content,
      showCancel: false,
      confirmText: "知道了",
      confirmColor: "#4A90D9"
    })
  },

  async onPullDownRefresh() {
    await this.loadNotices()
    wx.stopPullDownRefresh()
  }
})
