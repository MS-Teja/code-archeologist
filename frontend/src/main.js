import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import axios from 'axios'
import { store } from './store/analysis'

// Configure axios defaults
axios.defaults.withCredentials = true // Enable sending cookies
axios.defaults.baseURL = 'http://localhost:3000'

const app = createApp(App)
app.mount('#app')
