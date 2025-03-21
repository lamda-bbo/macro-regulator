// Copyright (c) 2018 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef TAO_PEGTL_INTERNAL_PEEK_MASK_UINT_HPP
#define TAO_PEGTL_INTERNAL_PEEK_MASK_UINT_HPP

#include <cstddef>
#include <cstdint>

#include "../config.hpp"

#include "input_pair.hpp"
#include "read_uint.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      namespace internal
      {
         template< typename R, typename R::type M >
         struct peek_mask_uint_impl
         {
            using data_t = typename R::type;
            using pair_t = input_pair< data_t >;

            template< typename Input >
            static pair_t peek( Input& in ) noexcept( noexcept( in.size( sizeof( data_t ) ) ) )
            {
               const std::size_t s = in.size( sizeof( data_t ) );
               if( s >= sizeof( data_t ) ) {
                  const data_t data = R::read( in.current() ) & M;
                  return { data, sizeof( data_t ) };
               }
               return { 0, 0 };
            }
         };

         template< std::uint16_t M >
         using peek_mask_uint16_be = peek_mask_uint_impl< read_uint16_be, M >;

         template< std::uint16_t M >
         using peek_mask_uint16_le = peek_mask_uint_impl< read_uint16_le, M >;

         template< std::uint32_t M >
         using peek_mask_uint32_be = peek_mask_uint_impl< read_uint32_be, M >;

         template< std::uint32_t M >
         using peek_mask_uint32_le = peek_mask_uint_impl< read_uint32_le, M >;

         template< std::uint64_t M >
         using peek_mask_uint64_be = peek_mask_uint_impl< read_uint64_be, M >;

         template< std::uint64_t M >
         using peek_mask_uint64_le = peek_mask_uint_impl< read_uint64_le, M >;

      }  // namespace internal

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#endif
