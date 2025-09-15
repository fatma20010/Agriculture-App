import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Define the user interface based on our database model
interface User {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
}

// Token data returned from authentication
interface TokenData {
  access_token: string;
  token_type: string;
  user_id: number;
  username: string;
  is_admin: boolean;
}

// Analysis records from database
interface AnalysisRecord {
  id: number;
  analysis_type: string;
  result_summary: {
    species: string;
    confidence: number;
    yellow_percentage: number;
    spot_count: number;
    estimated_compost_grams: number;
  };
  created_at: string;
}

const API_BASE_URL = 'http://localhost:5002';

const AuthExample = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [tokenData, setTokenData] = useState<TokenData | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [history, setHistory] = useState<AnalysisRecord[]>([]);
  
  // Form states
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  
  // Check if user is already logged in
  useEffect(() => {
    const storedToken = localStorage.getItem('auth_token');
    const storedUser = localStorage.getItem('user');
    
    if (storedToken && storedUser) {
      try {
        setTokenData(JSON.parse(storedToken));
        setUser(JSON.parse(storedUser));
        setIsLoggedIn(true);
        fetchUserHistory(JSON.parse(storedToken).access_token);
      } catch (e) {
        console.error('Failed to parse stored auth data', e);
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user');
      }
    }
  }, []);
  
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setMessage('');
    
    try {
      // For login, we need to use form data as per OAuth2 spec
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);
      formData.append('scope', 'user'); // Add 'admin' if the user needs admin access
      
      const response = await axios.post(`${API_BASE_URL}/token`, formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });
      
      setTokenData(response.data);
      localStorage.setItem('auth_token', JSON.stringify(response.data));
      
      // Now fetch user profile
      await fetchUserProfile(response.data.access_token);
      
      setIsLoggedIn(true);
      setMessage('Login successful!');
    } catch (err) {
      console.error('Login error:', err);
      setError('Login failed. Please check your credentials.');
    }
  };
  
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setMessage('');
    
    if (!username || !email || !password) {
      setError('Please fill all required fields');
      return;
    }
    
    try {
      const userData = {
        username,
        email,
        password,
        full_name: fullName || null,
        is_admin: false
      };
      
      const response = await axios.post(`${API_BASE_URL}/register`, userData);
      setMessage('Registration successful! Please login.');
      setIsRegistering(false);
    } catch (err: any) {
      console.error('Registration error:', err);
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
    }
  };
  
  const fetchUserProfile = async (token: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/users/me`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      setUser(response.data);
      localStorage.setItem('user', JSON.stringify(response.data));
      
      // After getting the user profile, fetch history
      fetchUserHistory(token);
    } catch (err) {
      console.error('Error fetching user profile:', err);
    }
  };
  
  const fetchUserHistory = async (token: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/user/history`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      setHistory(response.data);
    } catch (err) {
      console.error('Error fetching user history:', err);
    }
  };
  
  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
    setTokenData(null);
    setHistory([]);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
    setMessage('Logged out successfully');
  };
  
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };
  
  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || !tokenData) return;
    
    setMessage('Analyzing image...');
    setError('');
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${tokenData.access_token}`
        }
      });
      
      setAnalysisResults(response.data.results);
      setMessage('Analysis complete!');
      
      // Refresh history after analysis
      fetchUserHistory(tokenData.access_token);
    } catch (err: any) {
      console.error('Analysis error:', err);
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
    }
  };
  
  const handleGenerateReport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || !tokenData) return;
    
    setMessage('Generating report...');
    setError('');
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/generate_report`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${tokenData.access_token}`
        },
        responseType: 'blob'
      });
      
      // Create a URL for the blob
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'plant_health_report.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      setMessage('Report generated successfully!');
      
      // Refresh history after report generation
      fetchUserHistory(tokenData.access_token);
    } catch (err: any) {
      console.error('Report generation error:', err);
      setError('Report generation failed. Please try again.');
    }
  };
  
  // Render login/register form
  const renderAuthForms = () => (
    <div className="w-full max-w-md mx-auto bg-white p-8 rounded-lg shadow-md">
      <h2 className="text-2xl font-semibold mb-6 text-center">
        {isRegistering ? 'Create Account' : 'Login'}
      </h2>
      
      {message && (
        <div className="mb-4 p-3 bg-green-100 text-green-800 rounded-md">
          {message}
        </div>
      )}
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-800 rounded-md">
          {error}
        </div>
      )}
      
      {isRegistering ? (
        <form onSubmit={handleRegister} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Full Name (Optional)</label>
            <input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
              required
              minLength={8}
            />
          </div>
          
          <div className="flex justify-between">
            <button
              type="submit"
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
            >
              Register
            </button>
            <button
              type="button"
              onClick={() => setIsRegistering(false)}
              className="px-4 py-2 text-gray-600 rounded-md hover:bg-gray-100"
            >
              Back to Login
            </button>
          </div>
        </form>
      ) : (
        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
              required
            />
          </div>
          
          <div className="flex justify-between">
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Login
            </button>
            <button
              type="button"
              onClick={() => setIsRegistering(true)}
              className="px-4 py-2 text-gray-600 rounded-md hover:bg-gray-100"
            >
              Create Account
            </button>
          </div>
        </form>
      )}
    </div>
  );
  
  // Render authenticated user interface
  const renderUserDashboard = () => (
    <div className="w-full max-w-4xl mx-auto">
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold">Welcome, {user?.username}!</h2>
          <button
            onClick={handleLogout}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            Logout
          </button>
        </div>
        
        <div className="text-gray-700">
          <p><strong>Email:</strong> {user?.email}</p>
          {user?.full_name && <p><strong>Name:</strong> {user?.full_name}</p>}
          <p><strong>Role:</strong> {user?.is_admin ? 'Administrator' : 'User'}</p>
        </div>
      </div>
      
      {message && (
        <div className="mb-6 p-3 bg-green-100 text-green-800 rounded-md">
          {message}
        </div>
      )}
      
      {error && (
        <div className="mb-6 p-3 bg-red-100 text-red-800 rounded-md">
          {error}
        </div>
      )}
      
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <h3 className="text-xl font-semibold mb-4">Plant Analysis</h3>
        <form onSubmit={handleAnalyze} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Upload Image</label>
            <input
              type="file"
              onChange={handleFileSelect}
              className="mt-1 block w-full"
              accept="image/png, image/jpeg, image/jpg"
            />
          </div>
          
          <div className="flex space-x-4">
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              disabled={!selectedFile}
            >
              Analyze Plant
            </button>
            <button
              type="button"
              onClick={handleGenerateReport}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
              disabled={!selectedFile}
            >
              Generate Report
            </button>
          </div>
        </form>
      </div>
      
      {analysisResults && (
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h3 className="text-xl font-semibold mb-4">Analysis Results</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p><strong>Species:</strong> {analysisResults.species}</p>
              <p><strong>Confidence:</strong> {(analysisResults.confidence * 100).toFixed(2)}%</p>
              <p><strong>Yellow Percentage:</strong> {analysisResults.yellow_percentage.toFixed(2)}%</p>
              <p><strong>Spot Count:</strong> {analysisResults.spot_count}</p>
              <p><strong>Estimated Compost:</strong> {analysisResults.estimated_compost_grams.toFixed(2)}g</p>
            </div>
            <div className="grid grid-cols-1 gap-2">
              {analysisResults.original_image_path && (
                <img 
                  src={`${API_BASE_URL}/images/${analysisResults.original_image_path.split('/').pop()}`} 
                  alt="Original" 
                  className="w-full h-auto"
                />
              )}
            </div>
          </div>
        </div>
      )}
      
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-xl font-semibold mb-4">Analysis History</h3>
        {history.length === 0 ? (
          <p className="text-gray-500">No analysis history found.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Species</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Yellow %</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {history.map(record => (
                  <tr key={record.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(record.created_at).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {record.analysis_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {record.result_summary.species}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {record.result_summary.confidence.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {record.result_summary.yellow_percentage.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
  
  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4">
      <h1 className="text-3xl font-bold text-center text-green-800 mb-8">
        Plant Health Analysis with Secure Authentication
      </h1>
      {isLoggedIn ? renderUserDashboard() : renderAuthForms()}
    </div>
  );
};

export default AuthExample; 