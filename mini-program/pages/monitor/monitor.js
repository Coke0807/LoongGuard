var config = require("../../config")

Page({
  data: {
    // 状态
    loading: false,
    // 页面级无权限/未绑定提示（区别于单次拉流失败 streamError）
    hasPermission: true,
    permMsg: "",
    // 绑定幼儿列表（由 /api/wx/get-student-list 驱动）
    studentList: [],
    currentStudent: null,
    // 直播播放
    videoSrc: "",
    streamError: "",
    muted: true,
    currentTime: ""
  },

  onLoad: function() {
    this.updateTime()
    this.timer = setInterval(this.updateTime.bind(this), 1000)
  },

  onShow: function() {
    if (typeof this.getTabBar === 'function' && this.getTabBar()) {
      this.getTabBar().setData({ selected: 3 })
    }
    this.fetchStudents()
  },

  onUnload: function() {
    if (this.timer) {
      clearInterval(this.timer)
    }
  },

  updateTime: function() {
    var now = new Date()
    var pad = function(n) {
      return String(n).padStart(2, '0')
    }
    var time = now.getFullYear() + '-' +
      pad(now.getMonth() + 1) + '-' +
      pad(now.getDate()) + ' ' +
      pad(now.getHours()) + ':' +
      pad(now.getMinutes()) + ':' +
      pad(now.getSeconds())
    this.setData({ currentTime: time })
  },

  // ── 数据 ────────────────────────────────────────────────

  fetchStudents: function() {
    var that = this
    var request = require("../../utils/request")
    request("/get-student-list").then(function(res) {
      if (res.code !== 0 || !res.data || !res.data.length) {
        that.setData({
          hasPermission: false,
          permMsg: (res && res.msg) || "您尚未绑定幼儿，请联系幼儿园管理员录入信息后重试。",
          studentList: [],
          currentStudent: null,
          videoSrc: ""
        })
        return
      }
      var students = res.data.map(function(s) {
        return {
          id: s.student_id,
          name: s.name,
          className: s.class,
          switch: s.switch,
          statusText: s.switch === 1 ? "可观看" : "已关闭"
        }
      })
      var first = null
      for (var i = 0; i < students.length; i++) {
        if (students[i].switch === 1) { first = students[i]; break }
      }
      that.setData({ hasPermission: true, studentList: students })
      if (first) {
        that.setData({ currentStudent: first })
        that.loadStream()
      } else {
        that.setData({
          currentStudent: null,
          videoSrc: "",
          hasPermission: false,
          permMsg: "您绑定的幼儿均未开通远程观看权限，请联系幼儿园管理员开通。"
        })
      }
    }).catch(function() {
      that.setData({ hasPermission: false, permMsg: "网络异常，请检查后端服务后重试。" })
    })
  },

  switchStudent: function(e) {
    var id = e.currentTarget.dataset.id
    var list = this.data.studentList
    for (var i = 0; i < list.length; i++) {
      if (list[i].id === id) {
        this.setData({ currentStudent: list[i] })
        this.loadStream()
        return
      }
    }
  },

  // ── 拉流 ────────────────────────────────────────────────

  loadStream: function() {
    var that = this
    var s = this.data.currentStudent
    if (!s) { return }
    if (s.switch !== 1) {
      this.setData({
        videoSrc: "",
        hasPermission: false,
        permMsg: "该幼儿的远程观看权限已由园方关闭。"
      })
      return
    }
    this.setData({ loading: true, videoSrc: "", streamError: "" })
    var token = wx.getStorageSync('token') || ''
    wx.request({
      url: config.apiBaseUrl + "/stream/url",
      method: "GET",
      data: { token: token, student_id: s.id },
      success: function(res) {
        var body = res.data || {}
        if (res.statusCode === 403 || body.code !== 0) {
          that.setData({
            loading: false,
            streamError: body.msg || "获取直播流失败，请稍后重试"
          })
          return
        }
        // data.stream_url 已由后端签发带过期观看 token 的 m3u8 地址
        that.setData({ loading: false, videoSrc: body.data.stream_url, muted: true })
        that.videoCtx = wx.createVideoContext("monitorVideo", that)
        if (that.videoCtx) { that.videoCtx.mute(true) }
      },
      fail: function() {
        that.setData({ loading: false, streamError: "网络异常，无法连接直播服务" })
      }
    })
  },

  onVideoError: function() {
    this.setData({
      loading: false,
      streamError: "直播流连接失败，请点击「刷新」重试"
    })
  },

  // ── 播放器操作 ──────────────────────────────────────────

  toggleMute: function() {
    var next = !this.data.muted
    this.setData({ muted: next })
    if (!this.videoCtx) {
      this.videoCtx = wx.createVideoContext("monitorVideo", this)
    }
    if (this.videoCtx) { this.videoCtx.mute(next) }
  },

  refreshStream: function() {
    this.loadStream()
  },

  goFullscreen: function() {
    if (!this.videoCtx) {
      this.videoCtx = wx.createVideoContext("monitorVideo", this)
    }
    if (this.videoCtx) { this.videoCtx.requestFullScreen() }
  }
})
