import { useLive } from "./useLive.js";
import StatusBar from "./components/StatusBar.jsx";
import CameraCard from "./components/CameraCard.jsx";
import LastTomato from "./components/LastTomato.jsx";
import Stats from "./components/Stats.jsx";
import EventLog from "./components/EventLog.jsx";
import History from "./components/History.jsx";
import ChatPanel from "./components/ChatPanel.jsx";

export default function App() {
  const live = useLive();
  return (
    <>
      <StatusBar live={live} />
      <main className="grid">
        <CameraCard live={live} />
        <LastTomato tomato={live.tomatoes[0]} />
        <Stats stats={live.stats} />
        <ChatPanel />
        <EventLog events={live.events} />
        <History pageSize={25} refreshKey={live.newTomatoTick} statusKey={live.tomatoes[0]?.serial_status} />
      </main>
    </>
  );
}
