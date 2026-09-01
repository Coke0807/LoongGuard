const request = require("../../utils/request")
const config = require("../../config")

Page({
  data: {
    growthList: []
  },

  onLoad: function() {
    this.loadGrowthList()
  },

  onShow: function() {
    if (typeof this.getTabBar === 'function' && this.getTabBar()) {
      this.getTabBar().setData({ selected: 2 })
    }
  },

  loadGrowthList: function() {
    request("/get-growth-list").then(res => {
      if (res.code === 0 && res.data) {
        var list = res.data.map(function(item) {
          return {
            id: item.id,
            title: item.title,
            date: item.date,
            image: config.fileBaseUrl + item.image
          }
        })
        this.setData({ growthList: list })
      }
    }).catch(function(err) {
      console.error("加载成长记录失败", err)
    })
  },

  onPullDownRefresh: function() {
    this.loadGrowthList()
    wx.stopPullDownRefresh()
  },

  onPreviewImage: function(e) {
    var current = e.currentTarget.dataset.image
    var urls = []
    var list = this.data.growthList
    for (var i = 0; i < list.length; i++) {
      urls.push(list[i].image)
    }
    wx.previewImage({
      current: current,
      urls: urls
    })
  }
})
