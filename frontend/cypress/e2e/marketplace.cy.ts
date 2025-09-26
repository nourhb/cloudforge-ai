describe('Marketplace Page', () => {
  it('loads and shows headings', () => {
    cy.visit('/marketplace');
    cy.contains('API Marketplace').should('be.visible');
    cy.contains('Upload Worker').should('be.visible');
    cy.contains('Available Workers').should('be.visible');
  });
});
