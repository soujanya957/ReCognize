import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './pages/Login';
import UserInfo from './pages/UserInfo';
import Welcome from './pages/Welcome';
import MoCATest from './pages/MoCATest';
import Results from './pages/Results';
import Dashboard from './pages/Dashboard';
import CreateAccount from './pages/CreateAccount';

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<Welcome />} />
                <Route path="/login" element={<Login />} />
                <Route path='/create-account' element={<CreateAccount />} />

                <Route path="/userinfo" element={<UserInfo />} />
                <Route path="/mocatest" element={<MoCATest />} />
                <Route path="/results" element={<Results />} />
                <Route path="/dashboard" element={<Dashboard />} />
            </Routes>
        </Router>
    );
}

export default App;
