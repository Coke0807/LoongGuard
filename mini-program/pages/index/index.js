const request = require("../../utils/request")
const app = getApp()
Page({
  data:{studentList:[], loading:true, loginFailed:false},
  loginDone: false,

  async onLoad(){
    const result = await app.wxLogin()
    this.loginDone = true
    if(!result.success){
      this.setData({loading:false, loginFailed:true})
      return
    }
    if(!result.hasPhone){
      wx.redirectTo({url:"/pages/bindPhone/bindPhone"})
      return
    }
    await this.loadStudent()
  },

  retryLogin(){
    this.setData({loading:true, loginFailed:false})
    this.onLoad()
  },

  onShow(){
    if (typeof this.getTabBar === 'function' && this.getTabBar()) {
      this.getTabBar().setData({ selected: 0 })
    }
    if(this.loginDone && app.globalData.hasPhone){
      this.loadStudent()
    }
  },

  goBindPhone(){
    wx.navigateTo({url:"/pages/bindPhone/bindPhone"})
  },

  async loadStudent(){
    try{
      const res = await request("/get-student-list",{})
      if(res.code === 0){
        this.setData({studentList:res.data, loading:false})
      }else{
        this.setData({loading:false})
        wx.showToast({title:res.msg || "加载失败", icon:"none"})
      }
    }catch(e){
      this.setData({loading:false})
      wx.showToast({title:"网络异常", icon:"none"})
    }
  },

  unbindStudent(e){
    const studentId = e.currentTarget.dataset.id
    wx.showModal({
      title:"确认解除绑定",
      content:"解除后将无法查看该幼儿信息，确定要解除绑定吗？",
      success:async (res)=>{
        if(res.confirm){
          try{
            const result = await request("/unbind-student",{student_id:studentId})
            if(result.code === 0){
              wx.showToast({title:"解除绑定成功"})
              this.loadStudent()
            }else{
              wx.showToast({title:result.msg || "解除绑定失败",icon:"none"})
            }
          }catch(err){
            wx.showToast({title:"网络异常",icon:"none"})
          }
        }
      }
    })
  }
})
