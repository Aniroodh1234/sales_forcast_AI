import "./globals.css";

export const metadata = {
  title: "SalesCast AI | Enterprise Sales Forecasting Platform",
  description: "AI-powered sales forecasting system using SARIMA, Prophet, ETS, XGBoost, LSTM, and TFT models for accurate 8-week predictions across 43 US states.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
