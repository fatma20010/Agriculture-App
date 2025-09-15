import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const Index = () => {
  const [currentUser, setCurrentUser] = useState(null);
  const [currentPage, setCurrentPage] = useState('login');
  const [isAdminMode, setIsAdminMode] = useState(false);

  // Authentication states
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authSuccess, setAuthSuccess] = useState('');

  // Analysis states
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysisType, setAnalysisType] = useState('Farm');
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<any>(null);
  const [images, setImages] = useState<Array<{url: string, label: string}>>([]);
  const [error, setError] = useState<string | null>(null);
  const [reportUrl, setReportUrl] = useState<string | null>(null);

  // Shop states
  const [cart, setCart] = useState([]);
  const [products] = useState([
    { id: 1, name: 'Organic Compost', price: 25, category: 'Compost', currency: 'TND' },
    { id: 2, name: 'Plant Fertilizer', price: 15, category: 'Fertilizer', currency: 'TND' },
    { id: 3, name: 'Garden Tools Set', price: 45, category: 'Tools', currency: 'TND' },
    { id: 4, name: 'Indoor Plants', price: 20, category: 'Plants', currency: 'TND' }
  ]);

  useEffect(() => {
    const user = localStorage.getItem('currentUser');
    if (user) {
      setCurrentUser(user);
      setCurrentPage(user.startsWith('admin_') ? 'admin-dashboard' : 'home');
    }
  }, []);

  const login = () => {
    const accounts = JSON.parse(localStorage.getItem('accounts') || '{}');
    const accountKey = isAdminMode ? `admin_${username}` : username;
    
    if (accounts[accountKey] && accounts[accountKey] === password) {
      setCurrentUser(accountKey);
      localStorage.setItem('currentUser', accountKey);
      setCurrentPage(isAdminMode ? 'admin-dashboard' : 'home');
      setAuthError('');
    } else {
      setAuthError('Invalid credentials');
    }
  };

  const createAccount = () => {
    if (password !== confirmPassword) {
      setAuthError('Passwords do not match');
      return;
    }
    if (!username || !password) {
      setAuthError('Please fill all fields');
      return;
    }

    const accounts = JSON.parse(localStorage.getItem('accounts') || '{}');
    const accountKey = isAdminMode ? `admin_${username}` : username;
    
    if (accounts[accountKey]) {
      setAuthError('Account already exists');
      return;
    }

    accounts[accountKey] = password;
    localStorage.setItem('accounts', JSON.stringify(accounts));
    setAuthSuccess('Account created successfully');
    setTimeout(() => {
      setCurrentPage('login');
      setAuthSuccess('');
    }, 2000);
  };

  const logout = () => {
    localStorage.removeItem('currentUser');
    setCurrentUser(null);
    setCurrentPage('login');
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const validTypes = analysisType === 'Farm' 
        ? ['image/png', 'image/jpeg', 'image/jpg', 'image/mv2']
        : ['image/png', 'image/jpeg', 'image/jpg'];
      
      if (validTypes.includes(file.type)) {
        setSelectedFile(file);
        setError('');
      } else {
        setError(`Invalid file type for ${analysisType}. Please select a valid image.`);
      }
    }
  };

  const handleFarmAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    try {
      setLoading(true);
      setError('');
      console.log('Sending request to backend via proxy');
      
      const formData = new FormData();
      formData.append('image', selectedFile);
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error ${response.status}: ${errorText}`);
      }
      
      const data = await response.json();
      console.log('Analysis results:', data);
      
      if (data.status === 'success' && data.results) {
        setResults(data.results);
        
        // Prepare image URLs
        const imageUrls = [];
        if (data.results.original_image_path) {
          imageUrls.push({ 
            url: `/api/images/${data.results.original_image_path.split('/').pop()}`, 
            label: 'Original Image' 
          });
        }
        if (data.results.yellowing_mask_path) {
          imageUrls.push({ 
            url: `/api/images/${data.results.yellowing_mask_path.split('/').pop()}`, 
            label: 'Yellowing Detection' 
          });
        }
        if (data.results.edges_image_path) {
          imageUrls.push({ 
            url: `/api/images/${data.results.edges_image_path.split('/').pop()}`, 
            label: 'Spot Detection' 
          });
        }
        
        setImages(imageUrls);
        
        // Save to localStorage for diagnostics
        const analyses = JSON.parse(localStorage.getItem('analyses') || '[]');
        analyses.push({
          id: Date.now(),
          type: 'Farm',
          date: new Date().toISOString(),
          results: data.results,
          images: imageUrls
        });
        localStorage.setItem('analyses', JSON.stringify(analyses));
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (error) {
      console.error('Farm analysis error:', error);
      if (error.message.includes('Failed to fetch')) {
        setError('Connection failed. Please ensure the backend server is running on port 5002.');
      } else {
        setError(`Analysis failed: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGrassAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
      setLoading(true);
      setError('');
      console.log('Sending grass analysis request via proxy');
      
      const response = await fetch('/grass-api/analyze_grass', {
        method: 'POST',
        headers: { 
          'Accept': 'application/json'
        },
        body: formData
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log('Analysis results:', data);
      
      setResults(data.results);

      // Fetch output image
      const imageUrls = [];
      if (data.results.output_image_path) {
        const outputUrl = `/grass-api/images/${data.results.output_image_path.split('/').pop()}`;
        imageUrls.push({ url: outputUrl, label: 'Analysis Result' });
      }
      
      setImages(imageUrls);

      // Save to localStorage for diagnostics
      const analyses = JSON.parse(localStorage.getItem('analyses') || '[]');
      analyses.push({
        id: Date.now(),
        type: 'Hotel/Stadium',
        date: new Date().toISOString(),
        results: data.results,
        images: imageUrls
      });
      localStorage.setItem('analyses', JSON.stringify(analyses));

    } catch (error) {
      console.error('Grass analysis error:', error);
      if (error.message.includes('Failed to fetch')) {
        setError('Connection failed. Please ensure the grass analysis server is running on port 5000.');
      } else {
        setError(`Analysis failed: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = () => {
    if (analysisType === 'Farm') {
      handleFarmAnalyze();
    } else {
      handleGrassAnalyze();
    }
  };

  const handleFarmReport = async () => {
    if (!selectedFile || !results) {
      setError('Please analyze an image first');
      return;
    }

    try {
      setLoading(true);
      setError('');
      console.log('Generating report via proxy');
      
      const formData = new FormData();
      formData.append('image', selectedFile);
      
      const response = await fetch('/api/generate_report', {
        method: 'POST',
        body: formData
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error ${response.status}: ${errorText}`);
      }
      
      // Get the PDF blob
      const pdfBlob = await response.blob();
      
      // Create a URL for the blob
      const pdfUrl = URL.createObjectURL(pdfBlob);
      setReportUrl(pdfUrl);
      
      // Open the PDF in a new tab
      window.open(pdfUrl, '_blank');
      
      setError('');
    } catch (error) {
      console.error('Report generation error:', error);
      if (error.message.includes('Failed to fetch')) {
        setError('Connection failed. Please ensure the backend server is running on port 5002.');
      } else {
        setError(`Report generation failed: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGrassReport = async () => {
    if (!selectedFile || !results) {
      setError('Please analyze an image first');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
      setLoading(true);
      console.log('Generating grass report via proxy');
      
      const response = await fetch('/grass-api/generate_report', {
        method: 'POST',
        headers: { 'Accept': 'application/pdf' },
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error ${response.status}: ${errorText}`);
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/pdf')) {
        throw new Error('Invalid response format, expected PDF');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'grass_health_report.pdf';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

    } catch (error) {
      console.error('Report generation error:', error);
      if (error.message.includes('Failed to fetch')) {
        setError('Connection failed. Please ensure the grass analysis server is running on port 5000.');
      } else {
        setError(`Report generation failed: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = () => {
    if (analysisType === 'Farm') {
      handleFarmReport();
    } else {
      handleGrassReport();
    }
  };

  const renderLogin = () => (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-50 to-emerald-100 animate-fade-in">
      <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md">
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-6">
          {currentPage === 'login' ? 'Login' : 'Create Account'}
        </h2>
        
        <div className="mb-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={isAdminMode}
              onChange={(e) => setIsAdminMode(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm text-gray-600">Admin Account</span>
          </label>
        </div>

        <div className="space-y-4">
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
          />
          {currentPage === 'create-account' && (
            <input
              type="password"
              placeholder="Confirm Password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
            />
          )}
        </div>

        {authError && (
          <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg animate-fade-in">
            {authError}
          </div>
        )}

        {authSuccess && (
          <div className="mt-4 p-3 bg-green-100 border border-green-400 text-green-700 rounded-lg animate-fade-in">
            {authSuccess}
          </div>
        )}

        <div className="mt-6 space-y-3">
          <button
            onClick={currentPage === 'login' ? login : createAccount}
            className="w-full bg-green-500 hover:bg-green-600 text-white py-3 px-4 rounded-lg font-semibold transition-all duration-200 hover:scale-105 transform"
          >
            {currentPage === 'login' ? 'Login' : 'Create Account'}
          </button>
          
          <button
            onClick={() => setCurrentPage(currentPage === 'login' ? 'create-account' : 'login')}
            className="w-full text-green-600 hover:text-green-800 py-2 transition-colors duration-200"
          >
            {currentPage === 'login' ? 'Create Account' : 'Back to Login'}
          </button>
        </div>
      </div>
    </div>
  );

  const renderScanPlant = () => (
    <div className="space-y-6 animate-fade-in">
      <h2 className="text-2xl font-bold text-gray-800">Plant & Grass Analysis</h2>
      
      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Analysis Type</label>
            <select
              value={analysisType}
              onChange={(e) => {
                setAnalysisType(e.target.value);
                setResults(null);
                setImages([]);
                setError('');
              }}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
            >
              <option value="Farm">Farm (Plant Analysis)</option>
              <option value="Hotel/Stadium">Hotel/Stadium (Grass Analysis)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Upload Image</label>
            <input
              type="file"
              accept={analysisType === 'Farm' ? '.png,.jpg,.jpeg,.mv2' : '.png,.jpg,.jpeg'}
              onChange={handleFileSelect}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
            />
            <p className="text-sm text-gray-500 mt-1">
              Supported formats: {analysisType === 'Farm' ? 'PNG, JPG, JPEG, MV2' : 'PNG, JPG, JPEG'}
            </p>
          </div>

          {selectedFile && (
            <div className="p-3 bg-green-50 rounded-lg">
              <p className="text-sm text-green-700">
                Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            </div>
          )}

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg animate-fade-in">
              <p className="text-red-700">{error}</p>
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            className="w-full bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white py-3 px-4 rounded-lg font-semibold transition-all duration-200 hover:scale-105 transform disabled:transform-none disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Analyzing...
              </div>
            ) : (
              'Analyze Image'
            )}
          </button>
        </div>
      </div>

      {results && (
        <div className="bg-white p-6 rounded-lg shadow-md animate-slide-in">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">
            {analysisType === 'Farm' ? 'Analysis Results' : 'Grass Health Analysis'}
          </h3>
          {analysisType === 'Farm' ? (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <tbody className="divide-y divide-gray-200">
                  <tr>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">Species</td>
                    <td className="px-4 py-3 text-sm text-gray-700">{results.species || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">Confidence</td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {results.confidence ? (results.confidence * 100).toFixed(2) + '%' : 'N/A'}
                    </td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">Yellowing Percentage</td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {results.yellow_percentage ? results.yellow_percentage.toFixed(2) + '%' : 'N/A'}
                    </td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">Spot Count</td>
                    <td className="px-4 py-3 text-sm text-gray-700">{results.spot_count || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">Compost Needed</td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {results.estimated_compost_grams ? results.estimated_compost_grams.toFixed(2) + ' grams' : 'N/A'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <div className="text-lg text-gray-800 font-medium">
                  Healthy Grass Percentage: {typeof results.healthy_percentage === 'number' ? results.healthy_percentage.toFixed(2) : 'N/A'}%
                </div>
                <div className="text-lg text-gray-800 font-medium">
                  Unhealthy Grass Percentage: {typeof results.unhealthy_percentage === 'number' ? results.unhealthy_percentage.toFixed(2) : 'N/A'}%
                </div>
                <div className="text-lg text-gray-800 font-medium">
                  Compost Needed: {typeof results.compost_needed_grams === 'number' ? results.compost_needed_grams.toFixed(2) + ' g' : 'N/A'}
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-lg font-semibold text-gray-800 mb-2">Analysis Details</div>
                <div className="text-gray-800">{results.analysis_details || 'N/A'}</div>
              </div>
            </div>
          )}

          <button
            onClick={handleDownloadReport}
            disabled={loading}
            className="mt-4 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white py-2 px-4 rounded-lg font-semibold transition-all duration-200 hover:scale-105 transform disabled:transform-none disabled:cursor-not-allowed"
          >
            {loading ? 'Generating Report...' : 'Download PDF Report'}
          </button>
        </div>
      )}

      {images.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md animate-slide-in">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Analysis Images</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {images.map((image, index) => (
              <div key={index} className="text-center">
                <p className="text-sm font-medium text-gray-700 mb-2">{image.label}</p>
                <img
                  src={image.url}
                  alt={image.label}
                  className="w-full h-48 object-cover rounded-lg shadow-md"
                  onError={(e) => {
                    const target = e.currentTarget as HTMLImageElement;
                    target.src =
                      'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTgiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBOb3QgRm91bmQ8L3RleHQ+PC9zdmc+';
                    console.log(`Failed to load image: ${image.url}`);
                  }}
                />
                {/* Show grass percentages if available and this is a grass analysis */}
                {analysisType === 'Hotel/Stadium' && results && (
                  <div className="mt-2 text-sm text-gray-700">
                    <div>Healthy Grass Percentage: {results.healthy_percentage ?? results.healthy_percentage === 0 ? results.healthy_percentage : 'N/A'}%</div>
                    <div>Unhealthy Grass Percentage: {results.unhealthy_percentage ?? results.unhealthy_percentage === 0 ? results.unhealthy_percentage : 'N/A'}%</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderClientMenu = () => {
    const menuItems = [
      { key: 'home', label: 'Home', icon: '🏠' },
      { key: 'scan', label: 'Scan Plant', icon: '🔍' },
      { key: 'compost', label: 'Calculate Compost', icon: '🌱' },
      { key: 'shop', label: 'Shop', icon: '🛒' },
      { key: 'appointments', label: 'Appointments', icon: '📅' },
      { key: 'diagnostics', label: 'Diagnostics', icon: '📊' },
      { key: 'profile', label: 'Profile', icon: '👤' },
      { key: 'notifications', label: 'Notifications', icon: '🔔' },
      { key: 'assistance', label: 'Assistance', icon: '💬' },
    ];

    return (
      <div className="w-64 bg-green-800 text-white h-screen fixed left-0 top-0 overflow-y-auto">
        <div className="p-4">
          <h1 className="text-xl font-bold mb-6">Terra Verra</h1>
          <nav className="space-y-2">
            {menuItems.map((item) => (
              <button
                key={item.key}
                onClick={() => setCurrentPage(item.key)}
                className={`w-full text-left p-3 rounded-lg transition-colors duration-200 hover:bg-green-700 ${
                  currentPage === item.key ? 'bg-green-700' : ''
                }`}
              >
                <span className="mr-3">{item.icon}</span>
                {item.label}
              </button>
            ))}
          </nav>
          <button
            onClick={logout}
            className="w-full mt-6 p-3 bg-red-600 hover:bg-red-700 rounded-lg transition-colors duration-200"
          >
            Logout
          </button>
        </div>
      </div>
    );
  };

  const renderContent = () => {
    switch (currentPage) {
      case 'scan':
        return renderScanPlant();
      case 'home':
        return (
          <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Welcome to Terra Verra Dashboard</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Recent Analyses</h3>
                <p className="text-gray-600">{JSON.parse(localStorage.getItem('analyses') || '[]').length} analyses completed</p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Health Tips</h3>
                <p className="text-gray-600">Water plants every 3 days for optimal health</p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Cart Items</h3>
                <p className="text-gray-600">{cart.length} items in cart</p>
              </div>
            </div>
          </div>
        );
      case 'shop':
        return (
          <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Shop</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {products.map((product) => (
                <div key={product.id} className="bg-white p-6 rounded-lg shadow-md flex flex-col items-center">
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">{product.name}</h3>
                  <p className="text-green-600 font-bold text-xl">{product.price} TND</p>
                  <p className="text-gray-500 mb-4">{product.category}</p>
                  <button
                    onClick={() => setCart([...cart, product])}
                    className="mt-auto bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded-lg font-semibold transition-all duration-200 hover:scale-105 transform"
                  >
                    Add to Cart
                  </button>
                </div>
              ))}
            </div>
          </div>
        );
      default:
        return (
          <div className="animate-fade-in">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">{currentPage.charAt(0).toUpperCase() + currentPage.slice(1)}</h2>
            <div className="bg-white p-6 rounded-lg shadow-md">
              <p className="text-gray-600">This feature is coming soon!</p>
            </div>
          </div>
        );
    }
  };

  if (!currentUser) {
    return renderLogin();
  }

  return (
    <div className="flex">
      {renderClientMenu()}
      <div className="flex-1 ml-64 p-6 bg-gray-50 min-h-screen">
        {renderContent()}
      </div>
    </div>
  );
};

export default Index;