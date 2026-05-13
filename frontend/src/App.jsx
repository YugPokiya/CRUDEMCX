import React, { useEffect, useMemo, useState } from 'react';
const API = 'http://localhost:8080/api';
export default function App() {
  const [signal, setSignal] = useState(null);
  const [news, setNews] = useState([]);
  const [positions, setPositions] = useState([]);
  const [commodity, setCommodity] = useState('CRUDEOIL');
  const openPositions = useMemo(() => positions.filter((p) => p.status === 'OPEN'), [positions]);
  const closedPositions = useMemo(() => positions.filter((p) => p.status === 'CLOSED'), [positions]);
  async function refresh() {
    const [signalRes, newsRes, positionsRes] = await Promise.all([fetch(`${API}/signals/${commodity}`), fetch(`${API}/news`), fetch(`${API}/positions`)]);
    setSignal(await signalRes.json());
    setNews(await newsRes.json());
    setPositions(await positionsRes.json());
  }
  useEffect(() => { refresh().catch(console.error); }, [commodity]);
  async function exitPosition(id) {
    await fetch(`${API}/positions/${id}/exit`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ exit_price: signal?.entry ?? 0 }) });
    refresh();
  }
  return <main className="layout"><header className="header"><h1>MCX Smart Trade Dashboard</h1><select value={commodity} onChange={(e)=>setCommodity(e.target.value)}><option>CRUDEOIL</option><option>NATURALGAS</option><option>GOLD</option><option>SILVER</option></select></header>
  <section className="card"><h2>Reasoning Tab</h2>{signal && <><p><strong>{signal.commodity}</strong> → <span className="tag">{signal.side}</span></p><p>Entry: {signal.entry} | TP: {signal.take_profit} | SL: {signal.stop_loss}</p><p>Risk: {signal.risk_tag}</p><p>Seen reasoning: {signal.reason}</p><p>Unseen reasoning: News sentiment + sequence model confidence were blended.</p></>}</section>
  <section className="grid"><article className="card"><h2>Open Positions</h2>{openPositions.map((p)=><div key={p.id} className="position-row"><span>{p.commodity} {p.side} @ {p.entry_price}</span><button onClick={()=>exitPosition(p.id)}>Exit</button></div>)}{openPositions.length===0&&<p>No running position.</p>}</article>
  <article className="card"><h2>Closed Positions & PnL</h2>{closedPositions.map((p)=><div key={p.id} className="closed-row"><strong>{p.commodity}</strong><span>PnL: {p.pnl?.toFixed(2)}</span><small>Created: {p.order_created_at}</small><small>Finished: {p.order_finished_at}</small></div>)}{closedPositions.length===0&&<p>Closed history will appear here.</p>}</article></section>
  <section className="card"><h2>Validated Hourly News Notifications</h2>{news.map((n,idx)=><div key={idx} className="news-row"><h3>{n.title}</h3><p>{n.summary}</p><small>{n.source} | {n.action_hint} | validated: {String(n.validated)}</small></div>)}</section></main>;
}
