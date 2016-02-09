#include <sopt/types.h>
#include <sopt/utility.h>

int main(int, char const **) {

  sopt::Array<> input(6);
  input << 1e1, 2e1, 3e1, 4e1, 1e4, 2e4;

  if(not(sopt::soft_threshhold(input, 2.5e1).head(2).array() < 1e-8).all())
    throw std::exception();
  if(not(sopt::soft_threshhold(input, 2.5e1).tail(4).array() > 1e-8).all())
    throw std::exception();

  return 0;
}
