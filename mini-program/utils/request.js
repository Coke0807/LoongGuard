const config = require("../config")

function request(url, data, method="POST"){
    return new Promise((resolve,reject)=>{
        const token = wx.getStorageSync('token') || ''
        wx.request({
            url: config.apiBaseUrl + url,
            method,
            data,
            header:{
                "content-type":"application/json",
                ...(token ? {"Authorization": "Bearer " + token} : {})
            },
            success:res=>{
                if(res.statusCode === 401){
                    wx.removeStorageSync('token')
                    if(!request._isRelaunching){
                        request._isRelaunching = true
                        wx.reLaunch({
                            url:"/pages/index/index",
                            complete:()=>{ request._isRelaunching = false }
                        })
                    }
                    resolve({code:-1, msg:"登录失效，请重新登录", data:null})
                    return
                }
                resolve(res.data)
            },
            fail:err=>reject(err)
        })
    })
}
module.exports = request
