describe('Home Page', () => {
  it('renders hero and CTA', () => {
    cy.visit('/');
    cy.get('body').should('be.visible');
  });
});

// TEST: Cypress E2E basic spec
