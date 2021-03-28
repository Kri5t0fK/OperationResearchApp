# OperationResearchApp

Kitchen Manager App for Operational Research Class
Menedżer Kuchenny na Badania Operacyjne 2

## Notes

### Dimensions

- N is the number of recipes
- M is the number of ingredients

### R matrix

- i (the row index) is also the RecipeID
- R[ i ] is a vector containing ingredients
- R[ i ][ j ] is the amount of j-th ingredient (0 if not used)
- size is N x M

### X matrix

- X[ i ] is a vector containing a single solution
- X[ i ][ j ] is a boolean value indicating whether the given recipe will be used or not
- size is <number_of_solutions / size_of_solution_space> x N

### T vector

- T[ i ] is the preparation time for i-th recipe
- size is 1 x N

### Q vector

- Q[ j ] is the amount of j-th ingredient we have available
- size is 1 x M

### E vector

- E[ j ] is the expiration date of the j-th ingredient
- size is 1 x M

### P vector

- P[ j ] is the price of j-th ingredient
- size is 1 x M
