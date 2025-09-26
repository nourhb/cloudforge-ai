describe('Home Page', () => {
  it('renders hero and CTA', () => {
    cy.visit('http://localhost:3000');
    cy.contains('CloudForge AI').should('be.visible');
    cy.contains('Get Started').should('be.visible');
  });
});

// TEST: Cypress E2E basic spec
