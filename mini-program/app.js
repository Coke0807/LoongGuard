const request = require("./utils/request")
App({
  globalData:{
    token:"",
    hasPhone:false
  },
  async wxLogin(){
    try{
      const loginRes = await new Promise((resolve,reject)=>{
        wx.login({success:resolve, fail:reject})
      })
      const code = loginRes.code
      const res = await request("/login", {code})
      if(res.code !== 0){
        wx.showToast({title:"登录失败",icon:"none"})
        return {success:false, hasPhone:false}
      }
      const d = res.data
      this.globalData.token = d.token
      this.globalData.hasPhone = d.has_phone
      wx.setStorageSync('token', d.token)
      return {success:true, hasPhone:d.has_phone}
    }catch(e){
      wx.showToast({title:"网络异常",icon:"none"})
      return {success:false, hasPhone:false}
    }
  }
})
