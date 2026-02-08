export default function RootLayout({
    children,
  }: {
    children: JSX.Element | JSX.Element[];
  }) {
    return (
      <html lang="en">
        <body style={{ margin: 0, fontFamily: "system-ui, sans-serif" }}>
          {children}
        </body>
      </html>
    );
  }
  