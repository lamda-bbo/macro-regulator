// Copyright (c) 2017-2018 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef TAO_PEGTL_INTERNAL_BUMP_IMPL_HPP
#define TAO_PEGTL_INTERNAL_BUMP_IMPL_HPP

#include "../config.hpp"

#include "iterator.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      namespace internal
      {
         inline void bump( iterator& iter, const std::size_t count, const int ch ) noexcept
         {
            for( std::size_t i = 0; i < count; ++i ) {
               if( iter.data[ i ] == ch ) {
                  ++iter.line;
                  iter.byte_in_line = 0;
               }
               else {
                  ++iter.byte_in_line;
               }
            }
            iter.byte += count;
            iter.data += count;
         }

         inline void bump_in_this_line( iterator& iter, const std::size_t count ) noexcept
         {
            iter.data += count;
            iter.byte += count;
            iter.byte_in_line += count;
         }

         inline void bump_to_next_line( iterator& iter, const std::size_t count ) noexcept
         {
            ++iter.line;
            iter.byte += count;
            iter.byte_in_line = 0;
            iter.data += count;
         }

      }  // namespace internal

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#endif
