import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import AccountDetail from "./pages/AccountDetail";
import Dashboard from "./pages/Dashboard";
import Logs from "./pages/Logs";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="accounts/:key" element={<AccountDetail />} />
        <Route path="logs" element={<Logs />} />
      </Route>
    </Routes>
  );
}
