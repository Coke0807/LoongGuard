const request = require("../../utils/request")
const app = getApp()
Page({
  data:{
    phone:"",
    nickname:"",
    avatar:"",
    agreed: false,
    loading: false
  },
  inputPhone(e){
    this.setData({phone:e.detail.value})
  },
  inputNickname(e){
    this.setData({nickname:e.detail.value})
  },
  onChooseAvatar(e){
    this.setData({avatar:e.detail.avatarUrl})
  },
  toggleAgree(){
    this.setData({agreed: !this.data.agreed})
  },
  async submitBind(){
    const {phone, nickname, avatar, agreed} = this.data
    if(!agreed){
      return wx.showToast({title:"请先阅读并同意用户协议",icon:"none"})
    }
    if(!/^1[3-9]\d{9}$/.test(phone)){
      return wx.showToast({title:"请输入正确的手机号",icon:"none"})
    }
    this.setData({loading: true})
    try{
      const res = await request("/bind-phone",{phone, nickname, avatar})
      if(res.code === 0){
        wx.showToast({title:"绑定成功"})
        app.globalData.hasPhone = true
        setTimeout(()=>{
          wx.switchTab({url:"/pages/index/index"})
        },1000)
      }else{
        wx.showToast({title:res.msg || "绑定失败",icon:"none"})
      }
    }catch(e){
      wx.showToast({title:"网络异常，请检查网络或后端服务",icon:"none"})
    }finally{
      this.setData({loading: false})
    }
  }
})
