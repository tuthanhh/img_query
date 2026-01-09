import React from "react";
import { SearchProvider, useSearch } from "./context/SearchContext";
import { HomeView } from "./views/HomeView";
import { ResultsView } from "./views/ResultsView";

const Main: React.FC = () => {
  const { view } = useSearch();

  return (
    <>
      {view === "home" && <HomeView />}
      {view === "results" && <ResultsView />}
    </>
  );
};

const App: React.FC = () => {
  return (
    <SearchProvider>
      <Main />
    </SearchProvider>
  );
};

export default App;
