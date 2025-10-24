import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState('');
  const [answer, setAnswer] = useState('');
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState('');
  const [needsFeedback, setNeedsFeedback] = useState(false);
  const [critic, setCritic] = useState('');
  const [tools, setTools] = useState([]);
  const [iterations, setIterations] = useState(0);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);

  const solve = async () => {
    setStatus('Solving...');
    setAnswer('');
    setError('');
    setNeedsFeedback(false);
    try {
      const res = await axios.post('/api/solve', { query });
      setTaskId(res.data.task_id);
      pollStatus(res.data.task_id);
    } catch (err) {
      setStatus('Error');
      setError('Failed to start solving: ' + err.message);
    }
  };

  const pollStatus = (id) => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`/api/status/${id}`);
        const data = res.data;

        if (data.status === 'completed') {
          setAnswer(data.answer);
          setTools(data.tools || []);
          setIterations(data.iterations);
          setStatus(`Done in ${data.iterations} iteration(s)`);
          setError('');
          clearInterval(interval);
        } else if (data.status === 'needs_feedback') {
          setAnswer(data.answer);
          setCritic(data.critic_feedback);
          setNeedsFeedback(true);
          setError('');
          clearInterval(interval);
        } else if (data.status === 'error') {
          setError(data.answer || 'An error occurred');
          setStatus('Error');
          clearInterval(interval);
        } else if (data.status === 'processing') {
          setStatus('Processing...');
        } else {
          setStatus(data.status);
        }
      } catch (err) {
        setStatus('Error');
        setError('Failed to check status: ' + err.message);
        clearInterval(interval);
      }
    }, 1000);
  };

  const sendFeedback = async () => {
    setIsSubmittingFeedback(true);
    try {
      await axios.post('/api/feedback', {
        task_id: taskId,
        status: 'feedback_submitted',
        feedback: feedback || 'approve'
      });
      setNeedsFeedback(false);
      setFeedback('');
      setStatus('Refining...');
      pollStatus(taskId);
    } catch (err) {
      setError('Failed to submit feedback: ' + err.message);
      setStatus('Error');
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  return (
    <div className="container">
      <h1>Math Professor</h1>
      <p className="subtitle">Solve math problems with AI + Human Feedback</p>

      <textarea
        rows="4"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="e.g., A store sells apples for $2 and oranges for $3. You buy 5 fruits for $13. How many of each? Use substitution method."
        className="input"
      />

      <button onClick={solve} disabled={!query || status === 'Solving...'}>
        {status === 'Solving...' ? 'Solving...' : 'Solve'}
      </button>

      <p className="status"><strong>Status:</strong> {status}</p>

      {error && (
        <div className="error">
          <h3>Error:</h3>
          <pre>{error}</pre>
        </div>
      )}

      {answer && !error && (
        <div className="answer">
          <h3>Answer:</h3>
          <pre>{answer}</pre>
          <p><strong>Tools Used:</strong> {tools.join(', ') || 'None'}</p>
          <p><strong>Iterations:</strong> {iterations}</p>
        </div>
      )}

      {needsFeedback && (
        <div className="feedback-box">
          <h3>Critic Feedback</h3>
          <p>{critic}</p>
          <textarea
            rows="3"
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Type 'approve' or suggest improvement..."
            className="input"
          />
          <button 
            onClick={sendFeedback} 
            disabled={isSubmittingFeedback}
          >
            {isSubmittingFeedback ? 'Submitting...' : 'Submit Feedback'}
          </button>
        </div>
      )}
    </div>
  );
}

export default App;